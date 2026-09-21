#pragma once

#include <cstddef>
#include <vector>

// Выбор qreal должен совпадать у модуля и всех его потребителей.
#if defined(CONVERGENT_GRIDDING_STANDALONE_QREAL)
using qreal = double;
#elif defined(__has_include)
#  if __has_include(<QtCore/qglobal.h>)
#    include <QtCore/qglobal.h>
#  elif __has_include(<QtGlobal>)
#    include <QtGlobal>
#  else
#    error "QtGlobal was not found; define CONVERGENT_GRIDDING_STANDALONE_QREAL"
#  endif
#else
#  error "QtGlobal was not found; define CONVERGENT_GRIDDING_STANDALONE_QREAL"
#endif

namespace convergent {

// min/max — координаты крайних узлов; nx,ny >= 2, grid.size() == nx*ny.
// grid[iy*nx+ix]: слева направо, снизу вверх; grid[0] — (minx,miny).
// Входные значения задают начальный тренд и мягкий prior.
struct Surface {
    std::vector<qreal> grid;
    qreal minx{};
    qreal maxx{};
    qreal miny{};
    qreal maxy{};
    std::size_t nx{};
    std::size_t ny{};
};

// Конечные координаты внутри замкнутых границ Surface. Совпадающие x/y
// объединяются во взвешенное среднее; weight == 0 игнорируется, weight < 0 запрещен.
struct Point {
    qreal x{};
    qreal y{};
    qreal value{};
    qreal weight{qreal(1)};
};

// Последовательная XY-ломаная; value/weight не используются. Замыкание явно
// повторяет первую вершину. После удаления соседних повторов нужны две вершины.
using Fault = std::vector<Point>;

struct ConvergentGriddingOptions {
    // Число Snap-узлов масштабируется с шагом; на финале равно 1.
    std::size_t initialSnapNodes{16};
    // Желаемый предел интервалов по каждой оси; maxLevels ограничивает иерархию.
    std::size_t coarsestIntervals{8};
    std::size_t maxLevels{8};

    // Энергия: smoothness*||B*u||² + priorWeight*||u-prior||² + Snap/Taylor
    // и билинейные штрафы точек на финале. B — вторые разности со свободным краем.
    // priorWeight > 0 обеспечивает SPD; остальные силы могут быть нулевыми.
    qreal smoothness{qreal(1)};
    qreal priorWeight{qreal(1e-3)};
    qreal snapStrength{qreal(100)};
    qreal finalPointStrength{qreal(1000)};

    // Отдельная проекция C*u=d минимизирует узловую L2-поправку через Gram-PCG.
    bool enforceExactControls{true};
    std::size_t maxControlProjectionIterations{2000};
    // Порог: max(controlTolerance, 64*epsilon(qreal)*max(1,max|point.value|)).
    qreal controlTolerance{qreal(1e-10)};

    // Сигма Snap в ячейках текущего уровня.
    qreal gaussianSigma{qreal(1)};
    // 0: значение; 1: +градиент; 2: +Гессиан prior.
    int taylorOrder{2};
    // Делить объединенные веса на максимум, сохраняя их отношения.
    bool normalizePointWeights{true};

    // Matrix-free Jacobi-PCG: ||r|| <= absTol + relTol*max(||b||,||A*x0||).
    // Лимит на каждый уровень; при throwOnNonConvergence=false принят last iterate.
    std::size_t maxSolverIterations{1500};
    qreal relativeTolerance{qreal(1e-9)};
    qreal absoluteTolerance{qreal(0)};
    bool throwOnNonConvergence{true};
};

struct ConvergentGriddingLevelReport {
    std::size_t nx{};
    std::size_t ny{};
    // Номинальное число: разломы могут уменьшить фактическое число Snap-узлов.
    std::size_t snapNodes{};
    std::size_t solverIterations{};
    // ||r||/max(||b||,||A*x0||); истинная невязка пересчитывается раз в 50 шагов.
    qreal relativeResidual{};
    // Сходимость основного PCG, не точной проекции.
    bool converged{};
};

struct ConvergentGriddingReport {
    std::vector<ConvergentGriddingLevelReport> levels;
    // max|sampleSurface-value| в объединенных точках после преобразования в qreal.
    qreal maxControlError{};
    std::size_t controlProjectionIterations{};
    // В soft-режиме только диагностика относительно exact-допуска.
    bool controlsSatisfied{true};
};

// Проверка -> канонизация -> [Refine -> Snap/Taylor -> Smooth/PCG] ->
// optional exact projection -> проверка qreal -> commit. При исключении surface
// неизменна; без активных точек — побитовый no-op и пустой report.levels.
ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options = {});

// Разломы блокируют локальные связи; влияние может обойти свободный конец.
// Точки на разломе запрещены, узлы на нем сохраняют входные значения.
// Непредставимые грубые уровни пропускаются; на конечной сетке это ошибка.
// options обязателен для однозначности прежнего вызова (surface, points, {}).
ConvergentGriddingReport convergentGridding(
    Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options);

// Немутирующие перегрузки возвращают измененную копию поверхности.
Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const ConvergentGriddingOptions& options = {},
    ConvergentGriddingReport* report = nullptr);

Surface convergentGriddedSurface(
    const Surface& surface,
    const std::vector<Point>& points,
    const std::vector<Fault>& faults,
    const ConvergentGriddingOptions& options,
    ConvergentGriddingReport* report = nullptr);

// Билинейная выборка; с разломами — перенормировка взаимно видимых углов либо
// ближайший видимый узел в радиусе двух ячеек. Выборка на разломе — ошибка.
// Тот же оператор задает привязку точек. Surface не хранит разломы: передавайте
// те же faults при выборке и разрывайте связи через них при отрисовке.
qreal sampleSurface(
    const Surface& surface, qreal x, qreal y,
    const std::vector<Fault>& faults = {});

} // namespace convergent
