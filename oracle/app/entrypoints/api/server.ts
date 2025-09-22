import { createApp } from "./app";
import { Env } from "./env";

const app = createApp();
app.listen(Env.PORT, () => {
  console.log(`Pricing API listening on :${Env.PORT}`);
});
