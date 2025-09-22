import { IOdosClient } from "../../core/interfaces";
import { Address, ChainId, OdosQuoteRequest, OdosQuoteResponse } from "../../core/entities";
import { makeHttpClient } from "../http";
import { Env } from "../../entrypoints/api/env";

export class OdosClient implements IOdosClient {
  private http = makeHttpClient(Env.ODOS_BASE_URL);

  async getQuote(params: {
    chainId: ChainId;
    inputToken: Address;
    outputToken: Address;
    amount: string;
    userAddr: Address;
  }): Promise<OdosQuoteResponse> {
    const body: OdosQuoteRequest = {
      chainId: params.chainId,
      inputTokens: [{ tokenAddress: params.inputToken, amount: params.amount }],
      outputTokens: [{ tokenAddress: params.outputToken, proportion: 1 }],
      gasPrice: 0.02,
      userAddr: params.userAddr,
      slippageLimitPercent: 1.0,
      sourceBlacklist: [],
      sourceWhitelist: [],
      poolBlacklist: [],
      disableRFQs: true,
      referralCode: 0,
      compact: true,
      likeAsset: false,
      simple: false,
      pathViz: false
    };

    const { data } = await this.http.post<OdosQuoteResponse>("/sor/quote/v2", body);
    return data;
  }
}
