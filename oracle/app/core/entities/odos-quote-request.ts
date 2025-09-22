import type { Address } from "./address";
import type { ChainId } from "./chain-id";

export interface OdosQuoteRequest {
  chainId: ChainId;
  inputTokens: { tokenAddress: Address; amount: string }[];
  outputTokens: { tokenAddress: Address; proportion: number }[];
  gasPrice: number;
  userAddr: Address;
  slippageLimitPercent: number;
  sourceBlacklist: Address[];
  sourceWhitelist: Address[];
  poolBlacklist: Address[];
  disableRFQs: boolean;
  referralCode: number;
  compact: boolean;
  likeAsset: boolean;
  simple: boolean;
  pathViz: boolean;
}
