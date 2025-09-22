import express from "express";
import { createApiRouter } from "./router";
import { OdosClient } from "../../infrastructure/clients/odos-client";
import { EstimatorClient } from "../../infrastructure/clients/estimator-client";
import { PricingService } from "../../core/services/pricing-service";

export function createApp() {
  const app = express();
  app.use(express.json());

  const odos = new OdosClient();
  const estimator = new EstimatorClient();
  const pricing = new PricingService(odos, estimator);

  app.use("/api", createApiRouter(pricing));

  return app;
}
