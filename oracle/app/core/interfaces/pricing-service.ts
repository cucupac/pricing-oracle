import type { PricingRequest, PricingResult } from "../entities";

export interface IPricingService {
  price(req: PricingRequest): Promise<PricingResult>;
}
