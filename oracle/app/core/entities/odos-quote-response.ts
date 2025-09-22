import type { Address } from "./address";

export interface OdosQuoteResponse {
  inTokens: Address[];
  outTokens: Address[];
  inAmounts: string[];
  outAmounts: string[];
  gasEstimate: number;
  dataGasEstimate: number;
  gweiPerGas: number;
  gasEstimateValue: number;
  inValues: number[];
  outValues: number[];
  netOutValue: number;
  priceImpact: number;
  percentDiff: number;
  partnerFeePercent: number;
  blockNumber: number;
  pathId: string;
}
