import { IPricingService, IOdosClient, IEstimatorClient } from "../interfaces";
import { EstimatorRequest, PricingRequest, PricingResult } from "../entities";

const USDC_DECIMALS = 6;
const ETH_DECIMALS = 18;

function computeUsdcPerEth(amtInBase: string, amtOutBase: string): number {
  const inDec = Number(amtInBase) / 10 ** USDC_DECIMALS;
  const outDec = Number(amtOutBase) / 10 ** ETH_DECIMALS;
  if (outDec === 0) {
    return Infinity;
  }
  return (inDec / outDec) * 1e6;
}

export class PricingService implements IPricingService {
  constructor(
    private readonly odos: IOdosClient,
    private readonly estimator: IEstimatorClient
  ) {}

  async price(req: PricingRequest): Promise<PricingResult> {
    const quote = await this.odos.getQuote({
      chainId: req.chain_id,
      inputToken: req.input_token,
      outputToken: req.output_token,
      amount: req.input_token_amount,
      userAddr: process.env.ODOS_USER_ADDR as any
    });

    const inAmt = quote.inAmounts[0];
    const outAmt = quote.outAmounts[0];

    const priceRaw = computeUsdcPerEth(inAmt, outAmt);

    const estimatorPayload: EstimatorRequest = {
      vendor_id: "odos",
      chain_id: req.chain_id,
      input_token: req.input_token,
      output_token: req.output_token,
      input_amount: inAmt,
      output_amount_quote: outAmt
    };

    const amountInUsd = quote.inValues?.[0];
    if (typeof amountInUsd === "number" && Number.isFinite(amountInUsd)) {
      estimatorPayload.amount_in_usd = amountInUsd;
    }

    const outAmountScaled = Number(outAmt) / 10 ** ETH_DECIMALS;
    if (Number.isFinite(outAmountScaled)) {
      estimatorPayload.quote_out_units = outAmountScaled;
    }

    const gasUsd = quote.gasEstimateValue;
    if (typeof gasUsd === "number" && Number.isFinite(gasUsd)) {
      estimatorPayload.quote_gas_usd = gasUsd;
    }

    const adj = await this.estimator.getAdjustment(estimatorPayload);

    const priceAdjusted = priceRaw * adj.adjustment_factor;

    return {
      vendor: "odos",
      chain_id: req.chain_id,
      input_token: req.input_token,
      output_token: req.output_token,
      input_amount: inAmt,
      output_amount: outAmt,
      price_raw_usdc_per_eth: priceRaw,
      adjustment_factor: adj.adjustment_factor,
      price_adjusted_usdc_per_eth: priceAdjusted,
      pathId: quote.pathId,
      blockNumber: quote.blockNumber
    };
  }
}
