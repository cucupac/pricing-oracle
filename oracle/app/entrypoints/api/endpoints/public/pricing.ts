import { Router } from "express";
import { z } from "zod";
import { IPricingService } from "../../../../core/interfaces";
import { BadRequestError } from "../../../../core/errors";
import type { Address, PricingRequest } from "../../../../core/entities";

const addressSchema = z
  .string()
  .regex(/^0x[0-9a-fA-F]{40}$/)
  .transform((value) => value as Address);

const pricingRequestSchema = z.object({
  chain_id: z.number().int().positive(),
  input_token: addressSchema,
  output_token: addressSchema,
  input_token_amount: z.string().regex(/^[0-9]+$/)
});

export function registerPricingEndpoints(router: Router, pricingService: IPricingService): void {
  router.post("/price", async (req, res, next) => {
    try {
      const parsed = pricingRequestSchema.safeParse(req.body);
      if (!parsed.success) {
        throw new BadRequestError(parsed.error.message);
      }

      const payload: PricingRequest = parsed.data;
      const result = await pricingService.price(payload);
      res.json({ ok: true, result });
    } catch (err) {
      next(err);
    }
  });
}
