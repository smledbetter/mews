// High-performance parallel low-level parquet engine for LLM-under-AGENT query
// 2-thread parallel row-group processing with low-level ColumnReader API
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <arrow/io/file.h>
#include <parquet/api/reader.h>

static std::string extract_model_name(std::string_view json) {
    auto pos = json.find("\"llm\"");
    while (pos != std::string_view::npos) {
        size_t after = pos + 5;
        while (after < json.size() && (json[after]==' '||json[after]=='\t'||json[after]=='\n'||json[after]=='\r')) ++after;
        if (after < json.size() && json[after] == ':') {
            ++after;
            while (after < json.size() && (json[after]==' '||json[after]=='\t'||json[after]=='\n'||json[after]=='\r')) ++after;
            if (after < json.size() && json[after] == '{') {
                size_t brace_start = after;
                int depth = 1; size_t le = after + 1;
                while (le < json.size() && depth > 0) {
                    if (json[le]=='{') depth++; else if (json[le]=='}') depth--;
                    else if (json[le]=='"') { le++; while(le<json.size()&&json[le]!='"'){if(json[le]=='\\')le++;le++;} }
                    le++;
                }
                std::string_view llm_obj(json.data()+brace_start, le-brace_start);
                auto mn_pos = llm_obj.find("\"model_name\"");
                if (mn_pos != std::string_view::npos) {
                    size_t p = mn_pos + 12;
                    while(p<llm_obj.size()&&(llm_obj[p]==' '||llm_obj[p]=='\t'||llm_obj[p]=='\n'||llm_obj[p]=='\r'))++p;
                    if (p<llm_obj.size()&&llm_obj[p]==':') {
                        ++p;
                        while(p<llm_obj.size()&&(llm_obj[p]==' '||llm_obj[p]=='\t'||llm_obj[p]=='\n'||llm_obj[p]=='\r'))++p;
                        if (p<llm_obj.size()&&llm_obj[p]=='"') {
                            ++p; size_t start=p;
                            while(p<llm_obj.size()&&llm_obj[p]!='"'){if(llm_obj[p]=='\\')p++;p++;}
                            return std::string(llm_obj.substr(start, p-start));
                        }
                    }
                }
                return {};
            }
        }
        pos = json.find("\"llm\"", pos+1);
    }
    return {};
}

static double qcont(std::vector<double>& v, double q) {
    size_t n = v.size(); if (!n) return 0;
    double idx = q*(n-1); size_t lo=(size_t)idx; double f=idx-lo;
    if (lo+1>=n) return v[n-1];
    return v[lo]*(1-f)+v[lo+1]*f;
}

static void expand_ba(const int16_t* defs, const parquet::ByteArray* packed, int64_t nrows,
                      parquet::ByteArray* expanded, uint8_t* is_null) {
    int64_t vi = 0;
    for (int64_t i = 0; i < nrows; i++) {
        if (defs[i] > 0) { expanded[i] = packed[vi++]; is_null[i] = 0; }
        else { is_null[i] = 1; }
    }
}

static void expand_i64(const int16_t* defs, const int64_t* packed, int64_t nrows,
                       int64_t* expanded, uint8_t* is_null) {
    int64_t vi = 0;
    for (int64_t i = 0; i < nrows; i++) {
        if (defs[i] > 0) { expanded[i] = packed[vi++]; is_null[i] = 0; }
        else { is_null[i] = 1; }
    }
}

struct LocalResult {
    std::unordered_map<std::string, std::vector<double>> model_latencies;
};

int main(int argc, char* argv[]) {
    if (argc < 2) { std::cerr << "Usage: engine <tenant> [parquet_path]\n"; return 1; }
    std::string tenant_id = argv[1];
    const char* path = (argc >= 3) ? argv[2]
        : "/home/dev/projects/mews/data/adapters/wildchat/output/spans.parquet";

    // Prefetch file
    { int fd = open(path, O_RDONLY);
      struct stat st; fstat(fd, &st);
      posix_fadvise(fd, 0, st.st_size, POSIX_FADV_SEQUENTIAL | POSIX_FADV_WILLNEED);
      close(fd); }

    auto infile = *arrow::io::ReadableFile::Open(path);
    auto pq_base = parquet::ParquetFileReader::Open(infile);
    auto meta = pq_base->metadata();
    int num_rg = meta->num_row_groups();
    
    double ts_ms = 0.001;
    auto col4 = meta->schema()->Column(4);
    if (col4->logical_type() && col4->logical_type()->type() == parquet::LogicalType::Type::TIMESTAMP) {
        auto ts_type = std::static_pointer_cast<const parquet::TimestampLogicalType>(col4->logical_type());
        if (ts_type->time_unit() == parquet::LogicalType::TimeUnit::MILLIS) ts_ms = 1.0;
        else if (ts_type->time_unit() == parquet::LogicalType::TimeUnit::MICROS) ts_ms = 0.001;
        else if (ts_type->time_unit() == parquet::LogicalType::TimeUnit::NANOS) ts_ms = 1e-6;
    }
    
    int64_t max_rows = 0;
    for (int rg = 0; rg < num_rg; rg++)
        max_rows = std::max(max_rows, meta->RowGroup(rg)->num_rows());
    
    int nthreads = 2;
    std::atomic<int> next_rg{0};
    std::vector<LocalResult> results(nthreads);
    
    auto worker = [&](int tid) {
        auto inf = *arrow::io::ReadableFile::Open(path);
        auto pq = parquet::ParquetFileReader::Open(inf);
        auto& local = results[tid];
        
        std::vector<int16_t> defs(max_rows);
        std::vector<parquet::ByteArray> packed(max_rows);
        std::vector<parquet::ByteArray> sk_e(max_rows), tid_e(max_rows), sid_e(max_rows), pid_e(max_rows), attr_e(max_rows);
        std::vector<uint8_t> sk_n(max_rows), tid_n(max_rows), sid_n(max_rows), pid_n(max_rows), attr_n(max_rows);
        std::vector<int64_t> st_pk(max_rows), et_pk(max_rows), st_e(max_rows), et_e(max_rows);
        std::vector<uint8_t> st_n(max_rows), et_n(max_rows);
        
        while (true) {
            int rg = next_rg.fetch_add(1);
            if (rg >= num_rg) break;
            
            auto rg_r = pq->RowGroup(rg);
            int64_t N = meta->RowGroup(rg)->num_rows();
            int64_t vr;
            
            // All readers created up front to keep ByteArray memory alive
            auto r_sk   = std::static_pointer_cast<parquet::ByteArrayReader>(rg_r->Column(8));
            auto r_tid  = std::static_pointer_cast<parquet::ByteArrayReader>(rg_r->Column(12));
            auto r_sid  = std::static_pointer_cast<parquet::ByteArrayReader>(rg_r->Column(1));
            auto r_pid  = std::static_pointer_cast<parquet::ByteArrayReader>(rg_r->Column(2));
            auto r_st   = std::static_pointer_cast<parquet::Int64Reader>(rg_r->Column(4));
            auto r_et   = std::static_pointer_cast<parquet::Int64Reader>(rg_r->Column(5));
            auto r_attr = std::static_pointer_cast<parquet::ByteArrayReader>(rg_r->Column(11));
            
            // Read span_kind
            r_sk->ReadBatch(N, defs.data(), nullptr, packed.data(), &vr);
            expand_ba(defs.data(), packed.data(), N, sk_e.data(), sk_n.data());
            
            bool has_agent = false;
            for (int64_t i = 0; i < N; i++) {
                if (!sk_n[i] && sk_e[i].len == 5 && sk_e[i].ptr[0] == 'A') { has_agent = true; break; }
            }
            if (!has_agent) continue;
            
            // Read tenant_id
            r_tid->ReadBatch(N, defs.data(), nullptr, packed.data(), &vr);
            expand_ba(defs.data(), packed.data(), N, tid_e.data(), tid_n.data());
            
            bool has_match = false;
            for (int64_t i = 0; i < N; i++) {
                if (!sk_n[i] && sk_e[i].len == 3 && sk_e[i].ptr[0] == 'L' &&
                    !tid_n[i] && tid_e[i].len == (uint32_t)tenant_id.size() &&
                    memcmp(tid_e[i].ptr, tenant_id.data(), tid_e[i].len) == 0) {
                    has_match = true; break;
                }
            }
            if (!has_match) continue;
            
            // Read span_id + build AGENT set
            r_sid->ReadBatch(N, defs.data(), nullptr, packed.data(), &vr);
            expand_ba(defs.data(), packed.data(), N, sid_e.data(), sid_n.data());
            
            std::unordered_set<std::string_view> local_agents;
            for (int64_t i = 0; i < N; i++) {
                if (!sk_n[i] && sk_e[i].len == 5 && sk_e[i].ptr[0] == 'A' && !sid_n[i])
                    local_agents.emplace((const char*)sid_e[i].ptr, sid_e[i].len);
            }
            
            // Read parent_id
            r_pid->ReadBatch(N, defs.data(), nullptr, packed.data(), &vr);
            expand_ba(defs.data(), packed.data(), N, pid_e.data(), pid_n.data());
            
            // Identify qualifying rows
            std::vector<int64_t> qual;
            for (int64_t i = 0; i < N; i++) {
                if (sk_n[i] || sk_e[i].len != 3 || sk_e[i].ptr[0] != 'L') continue;
                if (tid_n[i] || tid_e[i].len != (uint32_t)tenant_id.size() ||
                    memcmp(tid_e[i].ptr, tenant_id.data(), tid_e[i].len) != 0) continue;
                if (pid_n[i]) continue;
                if (local_agents.find(std::string_view((const char*)pid_e[i].ptr, pid_e[i].len)) == local_agents.end()) continue;
                qual.push_back(i);
            }
            if (qual.empty()) continue;
            
            // Read timestamps
            r_st->ReadBatch(N, defs.data(), nullptr, st_pk.data(), &vr);
            expand_i64(defs.data(), st_pk.data(), N, st_e.data(), st_n.data());
            r_et->ReadBatch(N, defs.data(), nullptr, et_pk.data(), &vr);
            expand_i64(defs.data(), et_pk.data(), N, et_e.data(), et_n.data());
            
            // Read attributes
            r_attr->ReadBatch(N, defs.data(), nullptr, packed.data(), &vr);
            expand_ba(defs.data(), packed.data(), N, attr_e.data(), attr_n.data());
            
            for (int64_t idx : qual) {
                if (attr_n[idx] || st_n[idx] || et_n[idx]) continue;
                auto& av = attr_e[idx];
                std::string model = extract_model_name({(const char*)av.ptr, av.len});
                if (model.empty()) continue;
                double latency = (double)(et_e[idx] - st_e[idx]) * ts_ms;
                local.model_latencies[std::move(model)].push_back(latency);
            }
        }
    };
    
    std::vector<std::thread> threads;
    for (int t = 0; t < nthreads; t++) threads.emplace_back(worker, t);
    for (auto& t : threads) t.join();
    
    // Merge
    std::unordered_map<std::string, std::vector<double>> model_latencies;
    for (auto& lr : results) {
        for (auto& [m, v] : lr.model_latencies) {
            auto& d = model_latencies[m];
            d.insert(d.end(), v.begin(), v.end());
        }
    }
    
    struct Row { std::string model; int64_t n; double p50, p95; };
    std::vector<Row> rows;
    rows.reserve(model_latencies.size());
    for (auto& [m, v] : model_latencies) {
        std::sort(v.begin(), v.end());
        rows.push_back({m, (int64_t)v.size(), qcont(v, 0.5), qcont(v, 0.95)});
    }
    std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.p95 > b.p95; });
    
    std::ofstream out("result1.csv");
    out << "model_name,n,p50_ms,p95_ms\n";
    for (auto& r : rows) {
        out << '"';
        for (char c : r.model) { if(c=='"') out<<"\"\""; else out<<c; }
        out << '"' << ',' << r.n << ',' << std::to_string(r.p50) << ',' << std::to_string(r.p95) << '\n';
    }
    return 0;
}