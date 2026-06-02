import { z } from "zod";

export const ModelMetricsSchema = z.object({
  id: z.string(),
  name: z.enum(["Multi-Scale CNN", "Standard CNN", "KNN Baseline"]),
  mAP: z.number().min(0).max(1),
  precision: z.number().min(0).max(1),
  recall: z.number().min(0).max(1),
  f1Score: z.number().min(0).max(1),
  inferenceTimeMs: z.number().positive(),
  parametersCount: z.number().positive().optional(),
  description: z.string(),
});

export type ModelMetrics = z.infer<typeof ModelMetricsSchema>;

export const EpochLossSchema = z.object({
  epoch: z.number().int().positive(),
  multiScaleLoss: z.number().nonnegative(),
  standardLoss: z.number().nonnegative(),
});

export type EpochLoss = z.infer<typeof EpochLossSchema>;

export const DashboardDataSchema = z.object({
  metrics: z.array(ModelMetricsSchema),
  trainingHistory: z.array(EpochLossSchema),
});

export type DashboardData = z.infer<typeof DashboardDataSchema>;
