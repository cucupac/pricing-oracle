import type { Address } from "./address";
import type { ChainId } from "./chain-id";

export interface PricingRequest {
  chain_id: ChainId;
  input_token: Address;
  output_token: Address;
  input_token_amount: string;
}
