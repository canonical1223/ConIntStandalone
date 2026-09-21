#include "Form.h"
#include "ui_Form.h"

#include <QMetaType>
#include <cmath>
#include <limits>

namespace {
// Только преобразование типа. Ограничения алгоритма проверяет вызывающий модуль.
bool readCount(const QString& text, std::size_t& value)
{
    bool ok = false;
    const auto parsed = text.toULongLong(&ok, 10);
    if (!ok || text.trimmed().startsWith(QLatin1Char('-'))
        || parsed > std::numeric_limits<std::size_t>::max()) return false;
    value = static_cast<std::size_t>(parsed);
    return true;
}

bool readReal(const QString& text, qreal& value)
{
    bool ok = false;
    const double parsed = text.toDouble(&ok);
    const qreal converted = static_cast<qreal>(parsed);
    if (!ok || !std::isfinite(parsed) || !std::isfinite(static_cast<double>(converted))
        || (parsed != 0 && converted == 0)) return false;
    value = converted;
    return true;
}
} // namespace

Form::Form(QWidget* parent)
    : QWidget(parent), ui_(std::make_unique<Ui::Form>())
{
    qRegisterMetaType<std::size_t>("std::size_t");
    ui_->setupUi(this);
    // Учитываем начальные значения, измененные в Qt Designer.
    on_initialSnapNodes_textChanged(ui_->initialSnapNodes->text());
    on_coarsestIntervals_textChanged(ui_->coarsestIntervals->text());
    on_maxLevels_textChanged(ui_->maxLevels->text());
    on_smoothness_textChanged(ui_->smoothness->text());
    on_priorWeight_textChanged(ui_->priorWeight->text());
    on_snapStrength_textChanged(ui_->snapStrength->text());
    on_finalPointStrength_textChanged(ui_->finalPointStrength->text());
    on_enforceExactControls_toggled(ui_->enforceExactControls->isChecked());
    on_maxControlProjectionIterations_textChanged(ui_->maxControlProjectionIterations->text());
    on_controlTolerance_textChanged(ui_->controlTolerance->text());
    on_gaussianSigma_textChanged(ui_->gaussianSigma->text());
    on_taylorOrder_currentIndexChanged(ui_->taylorOrder->currentIndex());
    on_normalizePointWeights_toggled(ui_->normalizePointWeights->isChecked());
    on_maxSolverIterations_textChanged(ui_->maxSolverIterations->text());
    on_relativeTolerance_textChanged(ui_->relativeTolerance->text());
    on_absoluteTolerance_textChanged(ui_->absoluteTolerance->text());
    on_throwOnNonConvergence_toggled(ui_->throwOnNonConvergence->isChecked());
}

Form::~Form() = default;

void Form::on_initialSnapNodes_textChanged(const QString& text)
{
    std::size_t value;
    if (!readCount(text, value) || value == initialSnapNodes_) return;
    initialSnapNodes_ = value;
    emit initialSnapNodesChanged(value);
}

void Form::on_coarsestIntervals_textChanged(const QString& text)
{
    std::size_t value;
    if (!readCount(text, value) || value == coarsestIntervals_) return;
    coarsestIntervals_ = value;
    emit coarsestIntervalsChanged(value);
}

void Form::on_maxLevels_textChanged(const QString& text)
{
    std::size_t value;
    if (!readCount(text, value) || value == maxLevels_) return;
    maxLevels_ = value;
    emit maxLevelsChanged(value);
}

void Form::on_smoothness_textChanged(const QString& text)
{
    qreal value;
    if (!readReal(text, value) || value == smoothness_) return;
    smoothness_ = value;
    emit smoothnessChanged(value);
}

void Form::on_priorWeight_textChanged(const QString& text)
{
    qreal value;
    if (!readReal(text, value) || value == priorWeight_) return;
    priorWeight_ = value;
    emit priorWeightChanged(value);
}

void Form::on_snapStrength_textChanged(const QString& text)
{
    qreal value;
    if (!readReal(text, value) || value == snapStrength_) return;
    snapStrength_ = value;
    emit snapStrengthChanged(value);
}

void Form::on_finalPointStrength_textChanged(const QString& text)
{
    qreal value;
    if (!readReal(text, value) || value == finalPointStrength_) return;
    finalPointStrength_ = value;
    emit finalPointStrengthChanged(value);
}

void Form::on_enforceExactControls_toggled(bool value)
{
    if (value == enforceExactControls_) return;
    enforceExactControls_ = value;
    emit enforceExactControlsChanged(value);
}

void Form::on_maxControlProjectionIterations_textChanged(const QString& text)
{
    std::size_t value;
    if (!readCount(text, value) || value == maxControlProjectionIterations_) return;
    maxControlProjectionIterations_ = value;
    emit maxControlProjectionIterationsChanged(value);
}

void Form::on_controlTolerance_textChanged(const QString& text)
{
    qreal value;
    if (!readReal(text, value) || value == controlTolerance_) return;
    controlTolerance_ = value;
    emit controlToleranceChanged(value);
}

void Form::on_gaussianSigma_textChanged(const QString& text)
{
    qreal value;
    if (!readReal(text, value) || value == gaussianSigma_) return;
    gaussianSigma_ = value;
    emit gaussianSigmaChanged(value);
}

void Form::on_taylorOrder_currentIndexChanged(int value)
{
    if (value < 0 || value > 2 || value == taylorOrder_) return;
    taylorOrder_ = value;
    emit taylorOrderChanged(value);
}

void Form::on_normalizePointWeights_toggled(bool value)
{
    if (value == normalizePointWeights_) return;
    normalizePointWeights_ = value;
    emit normalizePointWeightsChanged(value);
}

void Form::on_maxSolverIterations_textChanged(const QString& text)
{
    std::size_t value;
    if (!readCount(text, value) || value == maxSolverIterations_) return;
    maxSolverIterations_ = value;
    emit maxSolverIterationsChanged(value);
}

void Form::on_relativeTolerance_textChanged(const QString& text)
{
    qreal value;
    if (!readReal(text, value) || value == relativeTolerance_) return;
    relativeTolerance_ = value;
    emit relativeToleranceChanged(value);
}

void Form::on_absoluteTolerance_textChanged(const QString& text)
{
    qreal value;
    if (!readReal(text, value) || value == absoluteTolerance_) return;
    absoluteTolerance_ = value;
    emit absoluteToleranceChanged(value);
}

void Form::on_throwOnNonConvergence_toggled(bool value)
{
    if (value == throwOnNonConvergence_) return;
    throwOnNonConvergence_ = value;
    emit throwOnNonConvergenceChanged(value);
}
