import type { Address } from "./address";
import type { ChainId } from "./chain-id";

export interface PricingResult {
  vendor: "odos";
  chain_id: ChainId;
  input_token: Address;
  output_token: Address;
  input_amount: string;
  output_amount: string;
  price_raw_usdc_per_eth: number;
  adjustment_factor: number;
  price_adjusted_usdc_per_eth: number;
  pathId: string;
  blockNumber: number;
}
