import { describe, it, expect, vi } from "vitest";
import { PricingService } from "../app/core/services/pricing-service";
import { IOdosClient, IEstimatorClient } from "../app/core/interfaces";

describe("PricingService", () => {
  it("computes adjusted price per spec formula", async () => {
    const odos: IOdosClient = {
      getQuote: vi.fn().mockResolvedValue({
        inTokens: [""],
        outTokens: [""],
        inAmounts: ["100000000"],
        outAmounts: ["20000000000000000"],
        gasEstimate: 0,
        dataGasEstimate: 0,
        gweiPerGas: 0.02,
        gasEstimateValue: 0,
        inValues: [0],
        outValues: [0],
        netOutValue: 0,
        priceImpact: 0,
        percentDiff: 0,
        partnerFeePercent: 0,
        blockNumber: 1,
        pathId: "pid"
      })
    };
    const estimator: IEstimatorClient = {
      getAdjustment: vi.fn().mockResolvedValue({ adjustment_factor: 0.9, model_version: "x" })
    };

    const svc = new PricingService(odos, estimator);
    const res = await svc.price({
      chain_id: 1,
      input_token: "0x0000000000000000000000000000000000000001",
      output_token: "0x0000000000000000000000000000000000000000",
      input_token_amount: "100000000"
    });

    expect(res.price_raw_usdc_per_eth).toBeCloseTo(5_000_000_000);
    expect(res.price_adjusted_usdc_per_eth).toBeCloseTo(4_500_000_000);
    expect(res.pathId).toBe("pid");
  });
});
