import { DashboardData } from "./schemas";

export const mockDashboardData: DashboardData = {
  metrics: [
    {
      id: "multi-scale-cnn",
      name: "Multi-Scale CNN",
      mAP: 0.88,
      precision: 0.91,
      recall: 0.85,
      f1Score: 0.87,
      inferenceTimeMs: 45.2,
      parametersCount: 45000000,
      description: "Mô hình đề xuất với Multi-scale Context Block tối ưu cho vật thể cực nhỏ.",
    },
    {
      id: "standard-cnn",
      name: "Standard CNN",
      mAP: 0.72,
      precision: 0.75,
      recall: 0.65,
      f1Score: 0.69,
      inferenceTimeMs: 38.5,
      parametersCount: 41000000,
      description: "Faster R-CNN nguyên thủy với ResNet50. Thường bỏ sót các vật thể dưới 10 pixel.",
    },
    {
      id: "knn-baseline",
      name: "KNN Baseline",
      mAP: 0.35,
      precision: 0.42,
      recall: 0.30,
      f1Score: 0.35,
      inferenceTimeMs: 250.0,
      description: "Phương pháp cổ điển dùng HOG + Sliding Window. Tốc độ rất chậm và độ chính xác thấp.",
    }
  ],
  trainingHistory: [
    { epoch: 1, multiScaleLoss: 1.5, standardLoss: 1.8 },
    { epoch: 2, multiScaleLoss: 1.1, standardLoss: 1.5 },
    { epoch: 3, multiScaleLoss: 0.8, standardLoss: 1.2 },
    { epoch: 4, multiScaleLoss: 0.6, standardLoss: 1.0 },
    { epoch: 5, multiScaleLoss: 0.45, standardLoss: 0.85 },
    { epoch: 6, multiScaleLoss: 0.35, standardLoss: 0.75 },
    { epoch: 7, multiScaleLoss: 0.28, standardLoss: 0.68 },
    { epoch: 8, multiScaleLoss: 0.22, standardLoss: 0.62 },
    { epoch: 9, multiScaleLoss: 0.18, standardLoss: 0.58 },
    { epoch: 10, multiScaleLoss: 0.15, standardLoss: 0.55 },
  ]
};
