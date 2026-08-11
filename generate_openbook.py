import os
import struct
import subprocess
import time

# 1. 3-карточный LUT (uint32_t)
def generate_3card_lut():
    lut = [0] * 2197
    for r1 in range(13):
        for r2 in range(13):
            for r3 in range(13):
                ranks = sorted([r1, r2, r3], reverse=True)
                key = r1 * 169 + r2 * 13 + r3
                mask = (1 << ranks[0]) | (1 << ranks[1]) | (1 << ranks[2])
                if ranks[0] == ranks[1] == ranks[2]:
                    val = (4 << 24) | (ranks[0] << 20) | mask
                elif ranks[0] == ranks[1]:
                    val = (2 << 24) | (ranks[0] << 20) | (ranks[2] << 16) | mask
                elif ranks[1] == ranks[2]:
                    val = (2 << 24) | (ranks[1] << 20) | (ranks[0] << 16) | mask
                else:
                    val = (1 << 24) | (ranks[0] << 20) | (ranks[1] << 16) | mask
                lut[key] = val
    return ",".join(map(str, lut))

lut_3card_str = generate_3card_lut()

# 2. C++/CUDA КОД V9.2 (УНИВЕРСАЛЬНЫЙ ATOMIC ADD DOUBLE)
cuda_code = f"""
#include <iostream>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>

constexpr int TOTAL_CANONICAL_HANDS = 134459;
constexpr int SIMS_PER_HAND = 2048; 
constexpr int CHECKPOINT_EVERY = 5000; 

#pragma pack(push, 1)
struct BookEntry {{
    uint32_t hand_key;
    uint8_t best_placement;
    float ev;
}};
#pragma pack(pop)

__constant__ uint32_t d_lut_3card[2197] = {{ {lut_3card_str} }};
__constant__ float FL_BONUS_QQ_KK_AA[3] = {{ 28.5f, 38.3f, 47.4f }};
__constant__ float FL_BONUS_TRIPS = 67.7f;

// Универсальный атомарный add для double (работает на ВСЕХ архитектурах CUDA)
__device__ inline double atomicAddDouble(double* address, double val) {{
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    }} while (assumed != old);
    return __longlong_as_double(old);
#endif
}}

__device__ inline uint32_t fast_rand(uint32_t* state) {{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}}

__device__ uint32_t eval_5card_bitboard(uint64_t mask) {{
    uint64_t spades = mask & 0x1FFF;
    uint64_t hearts = (mask >> 13) & 0x1FFF;
    uint64_t diamonds = (mask >> 26) & 0x1FFF;
    uint64_t clubs = (mask >> 39) & 0x1FFF;
    uint32_t ranks = spades | hearts | diamonds | clubs;
    bool flush = (mask == spades || mask == (hearts<<13) || mask == (diamonds<<26) || mask == (clubs<<39));
    bool straight = false;
    int str_high = -1;
    int ls = __clz(ranks & (ranks<<1) & (ranks<<2) & (ranks<<3) & (ranks<<4));
    if (ls < 32) {{ straight = true; str_high = 31 - ls; }}
    if (ranks == 0x100f) {{ straight = true; str_high = 3; }} 
    int quad=-1, trip=-1, pair1=-1, pair2=-1;
    for(int i=12; i>=0; --i) {{
        int c = ((spades>>i)&1) + ((hearts>>i)&1) + ((diamonds>>i)&1) + ((clubs>>i)&1);
        if(c==4) quad=i;
        else if(c==3) trip=i;
        else if(c==2) {{ if(pair1==-1) pair1=i; else pair2=i; }}
    }}
    if (straight && flush) return (9<<24) | (str_high<<20) | ranks;
    if (quad != -1) return (8<<24) | (quad<<20) | ranks;
    if (trip != -1 && pair1 != -1) return (7<<24) | (trip<<20) | (pair1<<16) | ranks;
    if (flush) return (6<<24) | ((31 - __clz(ranks))<<20) | ranks;
    if (straight) return (5<<24) | (str_high<<20) | ranks;
    if (trip != -1) return (4<<24) | (trip<<20) | ranks;
    if (pair1 != -1 && pair2 != -1) return (3<<24) | (pair1<<20) | (pair2<<16) | ranks;
    if (pair1 != -1) {{
        uint32_t k1 = 31 - __clz(ranks ^ (1 << pair1)); 
        return (2<<24) | (pair1<<20) | (k1<<16) | ranks;
    }}
    uint32_t r1 = 31 - __clz(ranks);
    uint32_t r2 = 31 - __clz(ranks ^ (1 << r1)); 
    return (1<<24) | (r1<<20) | (r2<<16) | ranks;
}}

__device__ double calc_progressive_score(const uint8_t* top, uint64_t mid_mask, uint64_t bot_mask) {{
    uint32_t r_top = d_lut_3card[(top[0]%13)*169 + (top[1]%13)*13 + (top[2]%13)];
    uint32_t r_mid = eval_5card_bitboard(mid_mask);
    uint32_t r_bot = eval_5card_bitboard(bot_mask);
    if (r_top > r_mid || r_mid > r_bot) return -6.0; 
    double score = 0.0;
    int bot_class = r_bot >> 24; int bot_rank = (r_bot >> 20) & 0xF;
    int mid_class = r_mid >> 24; int mid_rank = (r_mid >> 20) & 0xF;
    int top_class = r_top >> 24; int top_rank = (r_top >> 20) & 0xF;
    if      (bot_class == 9) score += (bot_rank == 12) ? 25.0 : 15.0;
    else if (bot_class == 8) score += 10.0;
    else if (bot_class == 7) score += 6.0;
    else if (bot_class == 6) score += 4.0;
    else if (bot_class == 5) score += 2.0;
    if      (mid_class == 9) score += (mid_rank == 12) ? 50.0 : 30.0;
    else if (mid_class == 8) score += 20.0;
    else if (mid_class == 7) score += 12.0;
    else if (mid_class == 6) score += 8.0;
    else if (mid_class == 5) score += 4.0;
    else if (mid_class == 4) score += 2.0;
    if (top_class == 4) {{ score += 10.0 + top_rank + FL_BONUS_TRIPS; }} 
    else if (top_class == 2) {{
        const float PAIR_ROY[13] = {{0,0,0,0, 1,2,3,4,5,6, 7,8,9}}; 
        score += PAIR_ROY[top_rank];
        if (top_rank == 10) score += FL_BONUS_QQ_KK_AA[0];
        else if (top_rank == 11) score += FL_BONUS_QQ_KK_AA[1];
        else if (top_rank == 12) score += FL_BONUS_QQ_KK_AA[2];
    }}
    score += (double)r_top * 1e-11 + (double)r_mid * 1e-14 + (double)r_bot * 1e-17;
    return score;
}}

__device__ float eval_partial(const uint8_t* top, int t, uint64_t mid_mask, int m, uint64_t bot_mask, int b) {{
    uint32_t r_top = (t == 3) ? d_lut_3card[(top[0]%13)*169 + (top[1]%13)*13 + (top[2]%13)] : 0;
    uint32_t r_mid = (m == 5) ? eval_5card_bitboard(mid_mask) : 0;
    uint32_t r_bot = (b == 5) ? eval_5card_bitboard(bot_mask) : 0;
    if (t == 3 && m == 5 && r_top > r_mid) return -1e9f;
    if (m == 5 && b == 5 && r_mid > r_bot) return -1e9f;
    if (t == 3 && b == 5 && r_top > r_bot) return -1e9f;
    return __popcll(bot_mask) * 0.1f;
}}

__device__ void play_street_greedy(uint8_t c1, uint8_t c2, uint8_t c3, uint8_t* top, int* t, uint64_t* mid_mask, int* m, uint64_t* bot_mask, int* b) {{
    uint8_t pairs[3][2] = {{ {{c1, c2}}, {{c1, c3}}, {{c2, c3}} }};
    float best_val = -1e10f;
    int best_p = 0, best_r1 = 0, best_r2 = 0;
    for (int p = 0; p < 3; ++p) {{
        uint8_t k1 = pairs[p][0]; uint8_t k2 = pairs[p][1];
        for (int r1 = 0; r1 < 3; ++r1) {{
            for (int r2 = 0; r2 < 3; ++r2) {{
                if (*t + (r1==0) + (r2==0) > 3) continue;
                if (*m + (r1==1) + (r2==1) > 5) continue;
                if (*b + (r1==2) + (r2==2) > 5) continue;
                uint8_t tmp_top[3]; for(int i=0; i<*t; ++i) tmp_top[i] = top[i];
                int tmp_t = *t; uint64_t tmp_m_mask = *mid_mask; int tmp_m = *m; uint64_t tmp_b_mask = *bot_mask; int tmp_b = *b;
                if (r1 == 0) tmp_top[tmp_t++] = k1; else if (r1 == 1) {{ tmp_m_mask |= (1ULL << ((k1/13)*13 + (k1%13))); tmp_m++; }} else {{ tmp_b_mask |= (1ULL << ((k1/13)*13 + (k1%13))); tmp_b++; }}
                if (r2 == 0) tmp_top[tmp_t++] = k2; else if (r2 == 1) {{ tmp_m_mask |= (1ULL << ((k2/13)*13 + (k2%13))); tmp_m++; }} else {{ tmp_b_mask |= (1ULL << ((k2/13)*13 + (k2%13))); tmp_b++; }}
                float val = eval_partial(tmp_top, tmp_t, tmp_m_mask, tmp_m, tmp_b_mask, tmp_b);
                if (val > best_val) {{ best_val = val; best_p = p; best_r1 = r1; best_r2 = r2; }}
            }}
        }}
    }}
    uint8_t k1 = pairs[best_p][0]; uint8_t k2 = pairs[best_p][1];
    if (best_r1 == 0) top[(*t)++] = k1; else if (best_r1 == 1) {{ *mid_mask |= (1ULL << ((k1/13)*13 + (k1%13))); (*m)++; }} else {{ *bot_mask |= (1ULL << ((k1/13)*13 + (k1%13))); (*b)++; }}
    if (best_r2 == 0) top[(*t)++] = k2; else if (best_r2 == 1) {{ *mid_mask |= (1ULL << ((k2/13)*13 + (k2%13))); (*m)++; }} else {{ *bot_mask |= (1ULL << ((k2/13)*13 + (k2%13))); (*b)++; }}
}}

__global__ void generate_book_kernel(const uint8_t* d_hands, BookEntry* d_book, int start_idx, int end_idx) {{
    int idx = blockIdx.x;
    int hand_idx = start_idx + idx;
    if (hand_idx >= end_idx) return;
    
    uint8_t my_cards[5];
    for(int i=0; i<5; ++i) my_cards[i] = d_hands[idx * 5 + i];
    
    extern __shared__ double s_evs[];
    int tid = threadIdx.x;
    
    for(int i=tid; i<243; i+=blockDim.x) s_evs[i] = 0.0;
    __syncthreads();

    uint32_t rng_state = 1337 + hand_idx * 256 + tid;
    int sims_per_thread = SIMS_PER_HAND / blockDim.x; 

    for (int c = 0; c < 243; ++c) {{
        int row_counts[3] = {{0, 0, 0}};
        uint8_t placement[5];
        int temp_c = c;
        for(int i=0; i<5; ++i) {{ placement[i] = temp_c % 3; row_counts[temp_c % 3]++; temp_c /= 3; }}
        
        if (row_counts[0] > 3 || row_counts[1] > 5 || row_counts[2] > 5) continue;
        
        double local_score = 0.0;
        
        for (int sim = 0; sim < sims_per_thread; ++sim) {{
            uint8_t deck[47];
            int d_idx = 0;
            for(int i=0; i<52; ++i) {{
                bool used = false;
                for(int j=0; j<5; ++j) if(my_cards[j] == i) used = true;
                if(!used) deck[d_idx++] = i;
            }}
            
            for(int i=0; i<12; ++i) {{
                uint32_t r = fast_rand(&rng_state);
                int swap_idx = i + (r % (47 - i));
                uint8_t tmp = deck[i]; deck[i] = deck[swap_idx]; deck[swap_idx] = tmp;
            }}
            
            uint8_t top[3]; uint64_t mid_mask = 0, bot_mask = 0;
            int t=0, m=0, b=0;
            
            for(int i=0; i<5; ++i) {{
                if(placement[i]==0) {{ top[t++] = my_cards[i]; }}
                else if(placement[i]==1) {{ mid_mask |= (1ULL << ((my_cards[i]/13)*13 + (my_cards[i]%13))); m++; }}
                else {{ bot_mask |= (1ULL << ((my_cards[i]/13)*13 + (my_cards[i]%13))); b++; }}
            }}
            
            int deal_idx = 0;
            for (int street = 0; street < 4; ++street) {{
                uint8_t c1 = deck[deal_idx++];
                uint8_t c2 = deck[deal_idx++];
                uint8_t c3 = deck[deal_idx++];
                play_street_greedy(c1, c2, c3, top, &t, &mid_mask, &m, &bot_mask, &b);
            }}
            
            local_score += calc_progressive_score(top, mid_mask, bot_mask);
        }}
        
        // ИСПРАВЛЕНО: Вызов нашей универсальной функции atomicAddDouble
        atomicAddDouble(&s_evs[c], local_score);
    }}
    __syncthreads();
    
    if (tid == 0) {{
        double max_ev = -1e8;
        uint8_t best_cfg = 0;
        for(int i=0; i<243; ++i) {{
            double final_ev = s_evs[i] / (double)SIMS_PER_HAND;
            if(final_ev > max_ev && s_evs[i] != 0.0) {{ max_ev = final_ev; best_cfg = i; }}
        }}
        
        uint32_t key = ((uint32_t)my_cards[0] << 24) | ((uint32_t)my_cards[1] << 18) | 
                       ((uint32_t)my_cards[2] << 12) | ((uint32_t)my_cards[3] << 6)  | 
                        (uint32_t)my_cards[4];

        d_book[idx].hand_key = key;
        d_book[idx].best_placement = best_cfg;
        d_book[idx].ev = (float)max_ev; 
    }}
}}

uint64_t get_canonical_hash(const uint8_t* hand) {{
    const uint8_t PERMS[24][4] = {{
        {{0,1,2,3}}, {{0,1,3,2}}, {{0,2,1,3}}, {{0,2,3,1}}, {{0,3,1,2}}, {{0,3,2,1}},
        {{1,0,2,3}}, {{1,0,3,2}}, {{1,2,0,3}}, {{1,2,3,0}}, {{1,3,0,2}}, {{1,3,2,0}},
        {{2,0,1,3}}, {{2,0,3,1}}, {{2,1,0,3}}, {{2,1,3,0}}, {{2,3,0,1}}, {{2,3,1,0}},
        {{3,0,1,2}}, {{3,0,2,1}}, {{3,1,0,2}}, {{3,1,2,0}}, {{3,2,0,1}}, {{3,2,1,0}}
    }};
    uint64_t min_hash = ~0ULL;
    for (int p = 0; p < 24; ++p) {{
        uint8_t mapped[5];
        for (int i = 0; i < 5; ++i) {{
            uint8_t rank = hand[i] % 13;
            uint8_t suit = hand[i] / 13;
            mapped[i] = (PERMS[p][suit] * 13) + rank;
        }}
        std::sort(mapped, mapped + 5, std::greater<uint8_t>());
        uint64_t hash = 0;
        for (int i = 0; i < 5; ++i) hash = (hash << 8) | mapped[i];
        if (hash < min_hash) min_hash = hash;
    }}
    return min_hash;
}}

int load_checkpoint(std::vector<BookEntry>& book) {{
    std::ifstream f("checkpoint_v9.bin", std::ios::binary);
    if (!f) return 0;
    uint32_t num_done = 0;
    f.read(reinterpret_cast<char*>(&num_done), sizeof(num_done));
    if (!f) return 0;
    if (num_done > book.size()) num_done = (uint32_t)book.size();
    f.read(reinterpret_cast<char*>(book.data()), num_done * sizeof(BookEntry));
    std::cout << "🔄 Восстановление из чекпоинта. Готово рук: " << num_done << std::endl;
    return (int)num_done;
}}

void save_checkpoint(const std::vector<BookEntry>& book, int num_done) {{
    std::ofstream f("checkpoint_v9.bin", std::ios::binary | std::ios::trunc);
    uint32_t n = (uint32_t)num_done;
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    f.write(reinterpret_cast<const char*>(book.data()), num_done * sizeof(BookEntry));
    f.flush();
}}

int main() {{
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    std::cout << "🚀 Обнаружено GPU: " << num_gpus << std::endl;

    std::vector<uint8_t> host_hands;
    std::unordered_set<uint64_t> seen_hashes;
    
    for(int a=0; a<48; ++a) {{
        for(int b=a+1; b<49; ++b) {{
            for(int c=b+1; c<50; ++c) {{
                for(int d=c+1; d<51; ++d) {{
                    for(int e=d+1; e<52; ++e) {{
                        uint8_t hand[5] = {{(uint8_t)a, (uint8_t)b, (uint8_t)c, (uint8_t)d, (uint8_t)e}};
                        uint64_t hash = get_canonical_hash(hand);
                        if (seen_hashes.insert(hash).second) {{
                            for(int i=0; i<5; ++i) host_hands.push_back(hand[i]);
                        }}
                    }}
                }}
            }}
        }}
    }}

    std::vector<BookEntry> final_book(TOTAL_CANONICAL_HANDS);
    int start_from = load_checkpoint(final_book);
    
    for (int chunk_start = start_from; chunk_start < TOTAL_CANONICAL_HANDS; chunk_start += CHECKPOINT_EVERY) {{
        int chunk_end = std::min(chunk_start + CHECKPOINT_EVERY, TOTAL_CANONICAL_HANDS);
        int chunk_size = chunk_end - chunk_start;

        #pragma omp parallel num_threads(num_gpus)
        {{
            int gpu_id = omp_get_thread_num();
            cudaSetDevice(gpu_id);

            int per_gpu = (chunk_size + num_gpus - 1) / num_gpus;
            int local_start = chunk_start + gpu_id * per_gpu;
            int local_end   = std::min(local_start + per_gpu, chunk_end);
            int local_count = local_end - local_start;

            if (local_count > 0) {{
                uint8_t* d_hands;
                BookEntry* d_book;
                cudaMalloc(&d_hands, local_count * 5);
                cudaMalloc(&d_book, local_count * sizeof(BookEntry));
                cudaMemcpy(d_hands, &host_hands[local_start * 5], local_count * 5, cudaMemcpyHostToDevice);

                // Запуск ядра
                generate_book_kernel<<<local_count, 256, 243 * sizeof(double)>>>(d_hands, d_book, local_start, local_end);
                cudaDeviceSynchronize();

                cudaMemcpy(&final_book[local_start], d_book, local_count * sizeof(BookEntry), cudaMemcpyDeviceToHost);
                cudaFree(d_hands); cudaFree(d_book);
            }}
        }}
        
        save_checkpoint(final_book, chunk_end);
        std::cout << "💾 Прогресс: " << chunk_end << " / " << TOTAL_CANONICAL_HANDS << " (" << (100.0 * chunk_end / TOTAL_CANONICAL_HANDS) << "%)" << std::endl;
        
        // ИСПРАВЛЕНО: Подавление warning через проверку условия
        if (system("git add checkpoint_v9.bin && git commit -m 'Auto-save checkpoint' && git push origin HEAD")) {{}}
    }}

    std::ofstream outfile("ofc_progressive_book_v9.bin", std::ios::binary);
    outfile.write(reinterpret_cast<const char*>(final_book.data()), final_book.size() * sizeof(BookEntry));
    outfile.close();

    std::cout << "✅ База V9.2 успешно сгенерирована!" << std::endl;
    return 0;
}}
"""

with open("generate_book.cu", "w") as f:
    f.write(cuda_code)

print("🔨 Компиляция CUDA кода V9.2 (с явным флагом -arch=sm_75)...")
# ИСПРАВЛЕНО: Добавлен флаг -arch=sm_75 для аппаратно точной компиляции под Tesla T4
subprocess.run("nvcc -O3 -arch=sm_75 -Xcompiler -fopenmp generate_book.cu -o generate_book", shell=True, check=True)

print("⚡ Запуск Multi-GPU генерации V9.2...")
subprocess.run("./generate_book", shell=True, check=True)

# --- АВТОМАТИЧЕСКИЙ ПУШ НА GITHUB ---
github_token = os.environ.get("GITHUB_TOKEN")
if github_token:
    print("🚀 Отправка результатов на GitHub...")
    subprocess.run("git config --global user.email 'kaggle-bot@example.com'", shell=True)
    subprocess.run("git config --global user.name 'Kaggle Bot'", shell=True)
    subprocess.run("git add ofc_progressive_book_v9.bin checkpoint_v9.bin", shell=True)
    subprocess.run("git commit -m 'Auto-generated Perfect Book V9.2 (Universal Double Atomics)'", shell=True)
    
    remote_url = subprocess.check_output("git config --get remote.origin.url", shell=True).decode().strip()
    if remote_url.startswith("https://"):
        auth_url = remote_url.replace("https://", f"https://oauth2:{github_token}@")
        subprocess.run(f"git remote set-url origin {auth_url}", shell=True)
        subprocess.run("git push origin HEAD", shell=True)
        print("✅ База V9.2 успешно загружена в репозиторий!")
