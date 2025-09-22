import type { EstimatorRequest, EstimatorResponse } from "../entities";

export interface IEstimatorClient {
  getAdjustment(body: EstimatorRequest): Promise<EstimatorResponse>;
}
