import type { Address, ChainId, OdosQuoteResponse } from "../entities";

export interface IOdosClient {
  getQuote(params: {
    chainId: ChainId;
    inputToken: Address;
    outputToken: Address;
    amount: string;
    userAddr: Address;
  }): Promise<OdosQuoteResponse>;
}
