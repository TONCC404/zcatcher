#ifndef LOSS_H
#define LOSS_H

#ifdef __cplusplus
extern "C" {
#endif

void launchCategoricalCrossEntropy(const float* pred, const float* target, float* loss, float* grad, int size);
void launchMSELoss(const float* pred, const float* target, float* loss, float* grad, int size);
void launchMAELoss(const float* pred, const float* target, float* loss, float* grad, int size);
void launchBinaryCrossEntropy(const float* pred, const float* target, float* loss, float* grad, int size);
void launchSmoothL1Loss(const float* pred, const float* target, float* loss, float* grad, int size);

#ifdef __cplusplus
}
#endif

#endif // LOSS_H