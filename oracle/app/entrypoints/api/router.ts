import { Router, type Request, type Response, type NextFunction } from "express";
import { IPricingService } from "../../core/interfaces";
import { registerPricingEndpoints } from "./endpoints/public/pricing";
import { registerHealthEndpoint } from "./endpoints/metrics/health";
import { BadRequestError, ExternalServiceError } from "../../core/errors";

export function createApiRouter(pricingService: IPricingService): Router {
  const router = Router();

  registerPricingEndpoints(router, pricingService);
  registerHealthEndpoint(router);

  router.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
    if (err instanceof BadRequestError) {
      return res.status(400).json({ ok: false, error: err.message });
    }
    if (err instanceof ExternalServiceError) {
      return res.status(err.status ?? 502).json({ ok: false, error: err.message });
    }
    console.error(err);
    return res.status(500).json({ ok: false, error: "internal_error" });
  });

  return router;
}
