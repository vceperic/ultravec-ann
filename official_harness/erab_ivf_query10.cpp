// Minimal output adapter for the pinned official RaBitQ-Library IVF implementation.
// Quantization, indexing, and search are provided by upstream. This adapter requests
// K neighbors and writes their identifiers as ivecs for the shared recall evaluator.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/utils/io.hpp"

using PID = rabitqlib::PID;
using data_type = rabitqlib::RowMajorArray<float>;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " index query.fvecs out.ivecs [nprobe] [K] [use_hacc:true|false]\n";
        return 1;
    }
    const char* index_file = argv[1];
    const char* query_file = argv[2];
    const char* out_file = argv[3];
    size_t nprobe = argc > 4 ? static_cast<size_t>(std::stoull(argv[4])) : 1000000;
    const size_t k = argc > 5 ? static_cast<size_t>(std::stoull(argv[5])) : 10;
    const bool use_hacc = argc > 6 && std::string(argv[6]) == "true";
    if (k == 0) {
        std::cerr << "K must be positive\n";
        return 2;
    }

    data_type query;
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::ivf::IVF ivf;
    ivf.load(index_file);
    nprobe = std::min(nprobe, ivf.num_clusters());

    std::ofstream output(out_file, std::ios::binary);
    if (!output) {
        std::cerr << "cannot open output: " << out_file << '\n';
        return 3;
    }
    std::vector<PID> results(k);
    for (size_t i = 0; i < query.rows(); ++i) {
        ivf.search(&query(i, 0), k, nprobe, results.data(), use_hacc);
        const auto width = static_cast<int32_t>(k);
        output.write(reinterpret_cast<const char*>(&width), sizeof(width));
        for (PID result : results) {
            const auto id = static_cast<int32_t>(result);
            output.write(reinterpret_cast<const char*>(&id), sizeof(id));
        }
    }
    if (!output) {
        std::cerr << "failed while writing output: " << out_file << '\n';
        return 4;
    }
    std::cerr << "wrote " << query.rows() << " x " << k << " identifiers with nprobe="
              << nprobe << " and use_hacc=" << (use_hacc ? "true" : "false") << '\n';
    return 0;
}
