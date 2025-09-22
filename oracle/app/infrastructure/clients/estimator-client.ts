import { IEstimatorClient } from "../../core/interfaces";
import { EstimatorRequest, EstimatorResponse } from "../../core/entities";
import { makeHttpClient } from "../http";
import { Env } from "../../entrypoints/api/env";

export class EstimatorClient implements IEstimatorClient {
  private http = makeHttpClient(Env.PREDICTOR_BASE_URL);

  async getAdjustment(body: EstimatorRequest): Promise<EstimatorResponse> {
    const { data } = await this.http.post<EstimatorResponse>("/predict", body);
    return data;
  }
}
