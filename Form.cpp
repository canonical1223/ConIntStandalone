#include "Form.h"
#include "ui_Form.h"

#include <QMetaType>

Form::Form(QWidget* parent)
    : QWidget(parent), ui_(std::make_unique<Ui::Form>())
{
    qRegisterMetaType<std::size_t>("std::size_t");
    ui_->setupUi(this);
    // Учитываем начальные значения, измененные в Qt Designer.
    on_initialSnapNodes_valueChanged(ui_->initialSnapNodes->value());
    on_coarsestIntervals_valueChanged(ui_->coarsestIntervals->value());
    on_maxLevels_valueChanged(ui_->maxLevels->value());
    on_smoothness_valueChanged(ui_->smoothness->value());
    on_priorWeight_valueChanged(ui_->priorWeight->value());
    on_snapStrength_valueChanged(ui_->snapStrength->value());
    on_finalPointStrength_valueChanged(ui_->finalPointStrength->value());
    on_enforceExactControls_toggled(ui_->enforceExactControls->isChecked());
    on_maxControlProjectionIterations_valueChanged(ui_->maxControlProjectionIterations->value());
    on_controlTolerance_valueChanged(ui_->controlTolerance->value());
    on_gaussianSigma_valueChanged(ui_->gaussianSigma->value());
    on_taylorOrder_currentIndexChanged(ui_->taylorOrder->currentIndex());
    on_normalizePointWeights_toggled(ui_->normalizePointWeights->isChecked());
    on_maxSolverIterations_valueChanged(ui_->maxSolverIterations->value());
    on_relativeTolerance_valueChanged(ui_->relativeTolerance->value());
    on_absoluteTolerance_valueChanged(ui_->absoluteTolerance->value());
    on_throwOnNonConvergence_toggled(ui_->throwOnNonConvergence->isChecked());
}

Form::~Form() = default;

void Form::on_initialSnapNodes_valueChanged(int value)
{
    if (value < 0) return;
    const auto converted = static_cast<std::size_t>(value);
    if (converted == initialSnapNodes_) return;
    initialSnapNodes_ = converted;
    emit initialSnapNodesChanged(initialSnapNodes_);
}

void Form::on_coarsestIntervals_valueChanged(int value)
{
    if (value < 0) return;
    const auto converted = static_cast<std::size_t>(value);
    if (converted == coarsestIntervals_) return;
    coarsestIntervals_ = converted;
    emit coarsestIntervalsChanged(coarsestIntervals_);
}

void Form::on_maxLevels_valueChanged(int value)
{
    if (value < 0) return;
    const auto converted = static_cast<std::size_t>(value);
    if (converted == maxLevels_) return;
    maxLevels_ = converted;
    emit maxLevelsChanged(maxLevels_);
}

void Form::on_smoothness_valueChanged(double value)
{
    const auto converted = static_cast<qreal>(value);
    if (converted == smoothness_) return;
    smoothness_ = converted;
    emit smoothnessChanged(smoothness_);
}

void Form::on_priorWeight_valueChanged(double value)
{
    const auto converted = static_cast<qreal>(value);
    if (converted == priorWeight_) return;
    priorWeight_ = converted;
    emit priorWeightChanged(priorWeight_);
}

void Form::on_snapStrength_valueChanged(double value)
{
    const auto converted = static_cast<qreal>(value);
    if (converted == snapStrength_) return;
    snapStrength_ = converted;
    emit snapStrengthChanged(snapStrength_);
}

void Form::on_finalPointStrength_valueChanged(double value)
{
    const auto converted = static_cast<qreal>(value);
    if (converted == finalPointStrength_) return;
    finalPointStrength_ = converted;
    emit finalPointStrengthChanged(finalPointStrength_);
}

void Form::on_enforceExactControls_toggled(bool value)
{
    if (value == enforceExactControls_) return;
    enforceExactControls_ = value;
    emit enforceExactControlsChanged(value);
}

void Form::on_maxControlProjectionIterations_valueChanged(int value)
{
    if (value < 0) return;
    const auto converted = static_cast<std::size_t>(value);
    if (converted == maxControlProjectionIterations_) return;
    maxControlProjectionIterations_ = converted;
    emit maxControlProjectionIterationsChanged(maxControlProjectionIterations_);
}

void Form::on_controlTolerance_valueChanged(double value)
{
    const auto converted = static_cast<qreal>(value);
    if (converted == controlTolerance_) return;
    controlTolerance_ = converted;
    emit controlToleranceChanged(controlTolerance_);
}

void Form::on_gaussianSigma_valueChanged(double value)
{
    const auto converted = static_cast<qreal>(value);
    if (converted == gaussianSigma_) return;
    gaussianSigma_ = converted;
    emit gaussianSigmaChanged(gaussianSigma_);
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

void Form::on_maxSolverIterations_valueChanged(int value)
{
    if (value < 0) return;
    const auto converted = static_cast<std::size_t>(value);
    if (converted == maxSolverIterations_) return;
    maxSolverIterations_ = converted;
    emit maxSolverIterationsChanged(maxSolverIterations_);
}

void Form::on_relativeTolerance_valueChanged(double value)
{
    const auto converted = static_cast<qreal>(value);
    if (converted == relativeTolerance_) return;
    relativeTolerance_ = converted;
    emit relativeToleranceChanged(relativeTolerance_);
}

void Form::on_absoluteTolerance_valueChanged(double value)
{
    const auto converted = static_cast<qreal>(value);
    if (converted == absoluteTolerance_) return;
    absoluteTolerance_ = converted;
    emit absoluteToleranceChanged(absoluteTolerance_);
}

void Form::on_throwOnNonConvergence_toggled(bool value)
{
    if (value == throwOnNonConvergence_) return;
    throwOnNonConvergence_ = value;
    emit throwOnNonConvergenceChanged(value);
}
