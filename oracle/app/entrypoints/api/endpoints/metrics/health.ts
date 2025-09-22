import { Router } from "express";
import { HealthStatusResponse } from "../../schemas/health";

export function registerHealthEndpoint(router: Router): void {
  router.get("/metrics/health", (_req, res) => {
    const payload: HealthStatusResponse = {
      status: "healthy",
      timestamp: new Date().toISOString()
    };

    res.json(payload);
  });
}
