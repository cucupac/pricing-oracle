import type { Address } from "./address";
import type { ChainId } from "./chain-id";

export interface EstimatorRequest {
  vendor_id: "odos";
  chain_id: ChainId;
  input_token: Address;
  output_token: Address;
  input_amount: string;
  output_amount_quote: string;
  amount_in_usd?: number;
  quote_out_units?: number;
  quote_gas_usd?: number;
  hop_count?: number;
  min_pool_tvl_usd?: number;
}
