#pragma once

#include "convergent_gridding.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Header-only search wrapper. Link the existing convergent_gridding library;
// this header does not contain a second implementation of the gridder.
namespace convergent::autotune {

struct MetricWeights {
    double mae{1};
    double rmse{0};
    double huber{0};
    double p95{0};
};

struct Metrics {
    std::size_t comparedNodes{};
    double mae{};
    double rmse{};
    double bias{};
    double huber{};
    double p95{};
    double maxAbsoluteError{};
};

// An empty field fixes that option at Config::baseline. Only modelling options
// are searched; solver tolerances and iteration caps remain in baseline.
struct SearchSpace {
    std::vector<std::size_t> initialSnapNodes;
    std::vector<std::size_t> coarsestIntervals;
    std::vector<std::size_t> maxLevels;
    std::vector<int> taylorOrder;
    std::vector<double> gaussianSigma;
    std::vector<double> smoothness;
    std::vector<double> priorWeight;
    std::vector<double> snapStrength;
    std::vector<double> finalPointStrength;
    std::vector<bool> enforceExactControls;
    std::vector<bool> normalizePointWeights;
};

inline SearchSpace defaultSearchSpace()
{
    SearchSpace space;
    space.initialSnapNodes = {4, 8, 16};
    space.coarsestIntervals = {4, 8, 16};
    space.taylorOrder = {0, 1, 2};
    space.gaussianSigma = {0.5, 1.0, 2.0};
    space.smoothness = {1.0};
    space.priorWeight = {1e-4, 1e-3, 1e-2};
    space.snapStrength = {10.0, 100.0, 1000.0};
    space.finalPointStrength = {100.0, 1000.0, 10000.0};
    // Exactness and weight semantics are kept fixed unless explicitly searched.
    return space;
}

struct Config {
    ConvergentGriddingOptions baseline{};
    MetricWeights metricWeights{};
    double huberDelta{1}; // Z units; only used when huber has positive weight
    std::size_t maxEvaluations{64};
    std::uint64_t seed{42};
};

struct Trial {
    ConvergentGriddingOptions options{};
    Metrics metrics{};
    double score{std::numeric_limits<double>::infinity()};
    double elapsedSeconds{};
    bool success{};
    std::string error;
    ConvergentGriddingReport report{};
};

struct Result {
    ConvergentGriddingOptions bestOptions{};
    Surface bestSurface{};
    Metrics bestMetrics{};
    double bestScore{std::numeric_limits<double>::infinity()};
    ConvergentGriddingReport bestReport{};
    std::size_t bestTrialIndex{};
    std::vector<Trial> trials;
};

namespace detail {

inline bool finite(double value) { return std::isfinite(value); }

inline void validateSurfaces(const Surface& input, const Surface& reference,
                             const std::vector<Point>& controls,
                             const std::vector<std::uint8_t>& mask,
                             const Config& config)
{
    if (input.nx < 2 || input.ny < 2 ||
        input.nx > std::numeric_limits<std::size_t>::max() / input.ny ||
        input.grid.size() != input.nx * input.ny) {
        throw std::invalid_argument("autotune: invalid input surface dimensions");
    }
    if (reference.nx != input.nx || reference.ny != input.ny ||
        reference.minx != input.minx || reference.maxx != input.maxx ||
        reference.miny != input.miny || reference.maxy != input.maxy ||
        reference.grid.size() != input.grid.size()) {
        throw std::invalid_argument("autotune: reference must use the input grid geometry");
    }
    if (!mask.empty() && mask.size() != input.grid.size()) {
        throw std::invalid_argument("autotune: reference mask has wrong size");
    }
    if (config.maxEvaluations == 0) {
        throw std::invalid_argument("autotune: maxEvaluations must be positive");
    }
    const auto& w = config.metricWeights;
    if (!finite(w.mae) || !finite(w.rmse) || !finite(w.huber) || !finite(w.p95) ||
        w.mae < 0 || w.rmse < 0 || w.huber < 0 || w.p95 < 0 ||
        !(w.mae + w.rmse + w.huber + w.p95 > 0)) {
        throw std::invalid_argument("autotune: metric weights must be nonnegative and nonzero");
    }
    if (w.huber > 0 && (!finite(config.huberDelta) || config.huberDelta <= 0)) {
        throw std::invalid_argument("autotune: huberDelta must be positive");
    }
    if (!finite(input.minx) || !finite(input.maxx) ||
        !finite(input.miny) || !finite(input.maxy) ||
        !(input.minx < input.maxx) || !(input.miny < input.maxy)) {
        throw std::invalid_argument("autotune: invalid grid bounds");
    }
    for (double value : input.grid) {
        if (!finite(value)) throw std::invalid_argument("autotune: input grid must be finite");
    }
    bool anyControl = false;
    for (const Point& point : controls) {
        if (point.weight > 0) anyControl = true;
    }
    if (!anyControl) {
        throw std::invalid_argument("autotune: at least one active control is required");
    }
}

// Matches the gridder's normalized orientation tolerance for nodes lying on a
// fault. Only the small bounding box of each segment is scanned, once per tune.
inline std::vector<std::size_t> comparisonNodes(
    const Surface& input, const Surface& reference,
    const std::vector<Fault>& faults, const std::vector<std::uint8_t>& mask)
{
    const long double ox = static_cast<long double>(input.minx);
    const long double oy = static_cast<long double>(input.miny);
    const long double scale = std::max(
        static_cast<long double>(input.maxx) - ox,
        static_cast<long double>(input.maxy) - oy);
    const long double tolerance = 64.0L *
        static_cast<long double>(std::numeric_limits<double>::epsilon());
    const long double physicalTolerance = tolerance * scale;
    std::vector<long double> xs(input.nx), ys(input.ny);
    // Grid::x/y in convergent_gridding.cpp use double arithmetic. Reproduce it
    // before normalizing, especially when world coordinates are very large.
    const double dx = (static_cast<double>(input.maxx) - static_cast<double>(input.minx)) /
        static_cast<double>(input.nx - 1);
    const double dy = (static_cast<double>(input.maxy) - static_cast<double>(input.miny)) /
        static_cast<double>(input.ny - 1);
    for (std::size_t i = 0; i < input.nx; ++i)
        xs[i] = i + 1 == input.nx ? static_cast<double>(input.maxx) :
            static_cast<double>(input.minx) + static_cast<double>(i) * dx;
    for (std::size_t j = 0; j < input.ny; ++j)
        ys[j] = j + 1 == input.ny ? static_cast<double>(input.maxy) :
            static_cast<double>(input.miny) + static_cast<double>(j) * dy;

    std::vector<std::uint8_t> excluded(input.grid.size(), 0);
    for (const Fault& fault : faults) {
        for (std::size_t k = 1; k < fault.size(); ++k) {
            const long double ax = fault[k - 1].x, ay = fault[k - 1].y;
            const long double bx = fault[k].x, by = fault[k].y;
            if (!std::isfinite(ax) || !std::isfinite(ay) ||
                !std::isfinite(bx) || !std::isfinite(by) ||
                (ax == bx && ay == by)) continue; // gridder validates faults
            const auto x0 = std::lower_bound(xs.begin(), xs.end(),
                std::min(ax, bx) - physicalTolerance);
            const auto x1 = std::upper_bound(xs.begin(), xs.end(),
                std::max(ax, bx) + physicalTolerance);
            const auto y0 = std::lower_bound(ys.begin(), ys.end(),
                std::min(ay, by) - physicalTolerance);
            const auto y1 = std::upper_bound(ys.begin(), ys.end(),
                std::max(ay, by) + physicalTolerance);
            const long double ndx = (bx - ax) / scale;
            const long double ndy = (by - ay) / scale;
            for (auto yi = y0; yi != y1; ++yi) {
                for (auto xi = x0; xi != x1; ++xi) {
                    const long double px = (*xi - ax) / scale;
                    const long double py = (*yi - ay) / scale;
                    const long double term1 = ndx * py;
                    const long double term2 = ndy * px;
                    const long double roundoff = 32.0L *
                        std::numeric_limits<long double>::epsilon() *
                        (std::abs(term1) + std::abs(term2));
                    if (std::abs(term1 - term2) <=
                        tolerance * std::max(std::abs(ndx), std::abs(ndy)) + roundoff) {
                        const std::size_t i = static_cast<std::size_t>(xi - xs.begin());
                        const std::size_t j = static_cast<std::size_t>(yi - ys.begin());
                        excluded[j * input.nx + i] = 1;
                    }
                }
            }
        }
    }

    std::vector<std::size_t> nodes;
    nodes.reserve(input.grid.size());
    for (std::size_t i = 0; i < input.grid.size(); ++i) {
        if (excluded[i] || (!mask.empty() && mask[i] == 0)) continue;
        if (!finite(reference.grid[i])) {
            throw std::invalid_argument("autotune: unmasked reference value is not finite");
        }
        nodes.push_back(i);
    }
    if (nodes.empty()) {
        throw std::invalid_argument("autotune: no comparable reference nodes");
    }
    return nodes;
}

inline Metrics measure(const Surface& candidate, const Surface& reference,
                       const std::vector<std::size_t>& nodes, double delta)
{
    Metrics m;
    m.comparedNodes = nodes.size();
    long double absoluteSum = 0, squaredSum = 0, signedSum = 0, huberSum = 0;
    std::vector<double> absoluteErrors;
    absoluteErrors.reserve(nodes.size());
    for (std::size_t i : nodes) {
        const long double error = static_cast<long double>(candidate.grid[i]) -
            static_cast<long double>(reference.grid[i]);
        const long double absolute = std::abs(error);
        if (!std::isfinite(error)) throw std::runtime_error("autotune: non-finite output error");
        absoluteSum += absolute;
        squaredSum += error * error;
        signedSum += error;
        const long double d = static_cast<long double>(delta);
        huberSum += absolute <= d ? absolute * absolute / (2 * d) : absolute - d / 2;
        absoluteErrors.push_back(static_cast<double>(absolute));
        m.maxAbsoluteError = std::max(m.maxAbsoluteError, static_cast<double>(absolute));
    }
    const long double n = static_cast<long double>(nodes.size());
    m.mae = static_cast<double>(absoluteSum / n);
    m.rmse = static_cast<double>(std::sqrt(squaredSum / n));
    m.bias = static_cast<double>(signedSum / n);
    m.huber = static_cast<double>(huberSum / n);
    const std::size_t p95Index = static_cast<std::size_t>(
        std::ceil(0.95L * n)) - 1;
    std::nth_element(absoluteErrors.begin(), absoluteErrors.begin() + p95Index,
                     absoluteErrors.end());
    m.p95 = absoluteErrors[p95Index];
    if (!finite(m.mae) || !finite(m.rmse) || !finite(m.huber) ||
        !finite(m.p95) || !finite(m.maxAbsoluteError)) {
        throw std::runtime_error("autotune: metric overflow");
    }
    return m;
}

inline double score(const Metrics& m, const MetricWeights& w)
{
    const long double sum = static_cast<long double>(w.mae) + w.rmse + w.huber + w.p95;
    const long double value = (static_cast<long double>(w.mae) * m.mae +
        static_cast<long double>(w.rmse) * m.rmse +
        static_cast<long double>(w.huber) * m.huber +
        static_cast<long double>(w.p95) * m.p95) / sum;
    const double result = static_cast<double>(value);
    if (!finite(result)) throw std::runtime_error("autotune: score overflow");
    return result;
}

inline bool sameTunableOptions(const ConvergentGriddingOptions& a,
                               const ConvergentGriddingOptions& b)
{
    return a.initialSnapNodes == b.initialSnapNodes &&
        a.coarsestIntervals == b.coarsestIntervals && a.maxLevels == b.maxLevels &&
        a.taylorOrder == b.taylorOrder &&
        a.gaussianSigma == b.gaussianSigma && a.smoothness == b.smoothness &&
        a.priorWeight == b.priorWeight && a.snapStrength == b.snapStrength &&
        a.finalPointStrength == b.finalPointStrength &&
        a.enforceExactControls == b.enforceExactControls &&
        a.normalizePointWeights == b.normalizePointWeights;
}

inline SearchSpace preparedSpace(SearchSpace s, const ConvergentGriddingOptions& b)
{
    if (s.initialSnapNodes.empty()) s.initialSnapNodes = {b.initialSnapNodes};
    if (s.coarsestIntervals.empty()) s.coarsestIntervals = {b.coarsestIntervals};
    if (s.maxLevels.empty()) s.maxLevels = {b.maxLevels};
    if (s.taylorOrder.empty()) s.taylorOrder = {b.taylorOrder};
    if (s.gaussianSigma.empty()) s.gaussianSigma = {b.gaussianSigma};
    if (s.smoothness.empty()) s.smoothness = {b.smoothness};
    if (s.priorWeight.empty()) s.priorWeight = {b.priorWeight};
    if (s.snapStrength.empty()) s.snapStrength = {b.snapStrength};
    if (s.finalPointStrength.empty()) s.finalPointStrength = {b.finalPointStrength};
    if (s.enforceExactControls.empty()) s.enforceExactControls = {b.enforceExactControls};
    if (s.normalizePointWeights.empty()) s.normalizePointWeights = {b.normalizePointWeights};
    return s;
}

inline std::array<std::size_t, 11> counts(const SearchSpace& s)
{
    return {s.initialSnapNodes.size(), s.coarsestIntervals.size(),
        s.maxLevels.size(), s.taylorOrder.size(), s.gaussianSigma.size(), s.smoothness.size(),
        s.priorWeight.size(), s.snapStrength.size(), s.finalPointStrength.size(),
        s.enforceExactControls.size(), s.normalizePointWeights.size()};
}

inline bool containsBaseline(const SearchSpace& s, const ConvergentGriddingOptions& b)
{
    const auto has = [](const auto& values, const auto& value) {
        return std::find(values.begin(), values.end(), value) != values.end();
    };
    return has(s.initialSnapNodes, b.initialSnapNodes) &&
        has(s.coarsestIntervals, b.coarsestIntervals) && has(s.maxLevels, b.maxLevels) &&
        has(s.taylorOrder, b.taylorOrder) && has(s.gaussianSigma, b.gaussianSigma) &&
        has(s.smoothness, b.smoothness) && has(s.priorWeight, b.priorWeight) &&
        has(s.snapStrength, b.snapStrength) &&
        has(s.finalPointStrength, b.finalPointStrength) &&
        has(s.enforceExactControls, b.enforceExactControls) &&
        has(s.normalizePointWeights, b.normalizePointWeights);
}

inline void assign(ConvergentGriddingOptions& o, const SearchSpace& s,
                   std::size_t dimension, std::size_t index)
{
    switch (dimension) {
    case 0: o.initialSnapNodes = s.initialSnapNodes[index]; break;
    case 1: o.coarsestIntervals = s.coarsestIntervals[index]; break;
    case 2: o.maxLevels = s.maxLevels[index]; break;
    case 3: o.taylorOrder = s.taylorOrder[index]; break;
    case 4: o.gaussianSigma = s.gaussianSigma[index]; break;
    case 5: o.smoothness = s.smoothness[index]; break;
    case 6: o.priorWeight = s.priorWeight[index]; break;
    case 7: o.snapStrength = s.snapStrength[index]; break;
    case 8: o.finalPointStrength = s.finalPointStrength[index]; break;
    case 9: o.enforceExactControls = s.enforceExactControls[index]; break;
    case 10: o.normalizePointWeights = s.normalizePointWeights[index]; break;
    }
}

inline ConvergentGriddingOptions fromIndex(const ConvergentGriddingOptions& base,
                                          const SearchSpace& s,
                                          const std::array<std::size_t, 11>& index)
{
    auto options = base;
    for (std::size_t d = 0; d < index.size(); ++d) assign(options, s, d, index[d]);
    return options;
}

} // namespace detail

// Reference and input must share exactly the same node coordinates. A zero in
// referenceMask excludes a cell; the mask may allow NaN in excluded reference
// cells. Grid nodes on fault geometry are always excluded automatically.
inline Result tune(const Surface& input, const Surface& reference,
                   const std::vector<Point>& controls,
                   const std::vector<Fault>& faults = {},
                   SearchSpace search = defaultSearchSpace(),
                   const Config& config = {},
                   const std::vector<std::uint8_t>& referenceMask = {})
{
    detail::validateSurfaces(input, reference, controls, referenceMask, config);
    const auto nodes = detail::comparisonNodes(input, reference, faults, referenceMask);
    search = detail::preparedSpace(std::move(search), config.baseline);
    const auto lengths = detail::counts(search);
    Result result;
    result.trials.reserve(config.maxEvaluations);

    const auto alreadyTried = [&](const ConvergentGriddingOptions& options) {
        return std::any_of(result.trials.begin(), result.trials.end(),
            [&](const Trial& t) { return detail::sameTunableOptions(t.options, options); });
    };
    const auto evaluate = [&](const ConvergentGriddingOptions& options) {
        if (result.trials.size() >= config.maxEvaluations || alreadyTried(options)) return false;
        Trial trial;
        trial.options = options;
        const auto started = std::chrono::steady_clock::now();
        try {
            ConvergentGriddingReport report;
            Surface surface = faults.empty()
                ? convergentGriddedSurface(input, controls, options, &report)
                : convergentGriddedSurface(input, controls, faults, options, &report);
            for (const auto& level : report.levels) {
                if (!level.converged) throw std::runtime_error("grid solver did not converge");
            }
            if (options.enforceExactControls && !report.controlsSatisfied) {
                throw std::runtime_error("exact controls were not satisfied");
            }
            trial.metrics = detail::measure(surface, reference, nodes,
                config.metricWeights.huber > 0 ? config.huberDelta : 1.0);
            trial.score = detail::score(trial.metrics, config.metricWeights);
            trial.report = report;
            trial.success = true;
            if (trial.score < result.bestScore) {
                result.bestScore = trial.score;
                result.bestMetrics = trial.metrics;
                result.bestOptions = options;
                result.bestSurface = std::move(surface);
                result.bestReport = std::move(report);
                result.bestTrialIndex = result.trials.size();
            }
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::exception& error) {
            trial.error = error.what();
        }
        trial.elapsedSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        result.trials.push_back(std::move(trial));
        return true;
    };

    evaluate(config.baseline);
    // Exhaustive search when the entire Cartesian grid fits the budget.
    std::size_t combinations = 1;
    bool fitsBudget = true;
    for (std::size_t length : lengths) {
        if (length > config.maxEvaluations / combinations) {
            fitsBudget = false;
            break;
        }
        combinations *= length;
    }
    const std::size_t baselineExtra =
        detail::containsBaseline(search, config.baseline) ? 0 : 1;
    if (fitsBudget && combinations <= config.maxEvaluations - baselineExtra) {
        std::array<std::size_t, 11> index{};
        for (std::size_t n = 0; n < combinations; ++n) {
            evaluate(detail::fromIndex(config.baseline, search, index));
            for (std::size_t d = 0; d < index.size(); ++d) {
                if (++index[d] < lengths[d]) break;
                index[d] = 0;
            }
        }
    } else {
        // First inspect isolated changes from baseline, then combinations.
        const std::size_t pilotLimit = std::max<std::size_t>(1, config.maxEvaluations / 2);
        for (std::size_t d = 0; d < lengths.size() && result.trials.size() < pilotLimit; ++d) {
            for (std::size_t i = 0; i < lengths[d] && result.trials.size() < pilotLimit; ++i) {
                auto options = config.baseline;
                detail::assign(options, search, d, i);
                evaluate(options);
            }
        }
        std::mt19937_64 random(config.seed);
        const std::size_t randomLimit = config.maxEvaluations - config.maxEvaluations / 3;
        std::size_t misses = 0;
        const std::size_t maxMisses = config.maxEvaluations >
            std::numeric_limits<std::size_t>::max() / 20
            ? std::numeric_limits<std::size_t>::max() : 20 * config.maxEvaluations;
        while (result.trials.size() < config.maxEvaluations &&
               (result.trials.size() < randomLimit ||
                !std::isfinite(static_cast<double>(result.bestScore))) &&
               misses < maxMisses) {
            std::array<std::size_t, 11> index{};
            for (std::size_t d = 0; d < index.size(); ++d) {
                std::uniform_int_distribution<std::size_t> draw(0, lengths[d] - 1);
                index[d] = draw(random);
            }
            if (evaluate(detail::fromIndex(config.baseline, search, index))) misses = 0;
            else ++misses;
        }
        // Coordinate refinement around the best successful candidate.
        bool progress = true;
        while (result.trials.size() < config.maxEvaluations && progress &&
               std::isfinite(static_cast<double>(result.bestScore))) {
            progress = false;
            for (std::size_t d = 0; d < lengths.size() &&
                 result.trials.size() < config.maxEvaluations; ++d) {
                for (std::size_t i = 0; i < lengths[d] &&
                     result.trials.size() < config.maxEvaluations; ++i) {
                    auto options = result.bestOptions;
                    detail::assign(options, search, d, i);
                    if (evaluate(options)) progress = true;
                }
            }
        }
    }
    if (!std::isfinite(static_cast<double>(result.bestScore))) {
        const std::string cause = result.trials.empty() ? "no trials" : result.trials.back().error;
        throw std::runtime_error("autotune: all candidates failed; last error: " + cause);
    }
    return result;
}

// The text is also valid C++ option assignments after declaring `options`.
// All fields are printed, including fixed numerical solver settings.
inline std::string formatBestOptions(const Result& result)
{
    if (!std::isfinite(static_cast<double>(result.bestScore))) {
        throw std::invalid_argument("autotune: result has no successful candidate");
    }
    const auto& o = result.bestOptions;
    const auto& m = result.bestMetrics;
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::boolalpha
        << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "// Best convergent_gridding options\n"
        << "// score = " << result.bestScore << ", MAE = " << m.mae
        << ", RMSE = " << m.rmse << ", P95 = " << m.p95 << "\n"
        << "// bias = " << m.bias << ", Huber = " << m.huber
        << ", maxAbsoluteError = " << m.maxAbsoluteError << "\n"
        << "// comparedNodes = " << m.comparedNodes
        << ", evaluatedCandidates = " << result.trials.size() << "\n"
        << "convergent::ConvergentGriddingOptions options;\n"
        << "options.initialSnapNodes = " << o.initialSnapNodes << ";\n"
        << "options.coarsestIntervals = " << o.coarsestIntervals << ";\n"
        << "options.maxLevels = " << o.maxLevels << ";\n"
        << "options.smoothness = " << o.smoothness << ";\n"
        << "options.priorWeight = " << o.priorWeight << ";\n"
        << "options.snapStrength = " << o.snapStrength << ";\n"
        << "options.finalPointStrength = " << o.finalPointStrength << ";\n"
        << "options.enforceExactControls = " << o.enforceExactControls << ";\n"
        << "options.maxControlProjectionIterations = "
        << o.maxControlProjectionIterations << ";\n"
        << "options.controlTolerance = " << o.controlTolerance << ";\n"
        << "options.gaussianSigma = " << o.gaussianSigma << ";\n"
        << "options.taylorOrder = " << o.taylorOrder << ";\n"
        << "options.normalizePointWeights = " << o.normalizePointWeights << ";\n"
        << "options.maxSolverIterations = " << o.maxSolverIterations << ";\n"
        << "options.relativeTolerance = " << o.relativeTolerance << ";\n"
        << "options.absoluteTolerance = " << o.absoluteTolerance << ";\n"
        << "options.throwOnNonConvergence = " << o.throwOnNonConvergence << ";\n";
    return out.str();
}

inline void saveBestOptions(const Result& result,
                            const std::filesystem::path& filePath)
{
    if (filePath.empty()) {
        throw std::invalid_argument("autotune: report file path is empty");
    }
    std::ofstream file(filePath, std::ios::binary | std::ios::trunc);
    if (!file) {
        throw std::runtime_error("autotune: cannot open the report file");
    }
    file << formatBestOptions(result);
    file.flush();
    if (!file) {
        throw std::runtime_error("autotune: failed to write the report file");
    }
}

// Convenience entry point: tune, save the exact parameter text, then print it.
// Pass an absolute path on C: to make the destination independent of cwd.
inline Result tuneAndReport(
    const Surface& input, const Surface& reference,
    const std::vector<Point>& controls,
    const std::filesystem::path& reportFileOnC,
    const std::vector<Fault>& faults = {},
    SearchSpace search = defaultSearchSpace(),
    const Config& config = {},
    const std::vector<std::uint8_t>& referenceMask = {})
{
    Result result = tune(input, reference, controls, faults,
                         std::move(search), config, referenceMask);
    saveBestOptions(result, reportFileOnC);
    std::cout << formatBestOptions(result)
              << "Saved to: " << reportFileOnC.string() << std::endl;
    return result;
}

} // namespace convergent::autotune
