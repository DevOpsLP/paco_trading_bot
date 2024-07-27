require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const axios = require('axios');
const crypto = require('crypto');

const app = express();
app.use(bodyParser.json());

const apiKey = process.env.BITGET_API_KEY;
const secretKey = process.env.BITGET_SECRET_KEY;
const passphrase = process.env.BITGET_PASSPHRASE;

const getTimestamp = () => {
  return Date.now();
};

const signMessage = (message, secretKey) => {
  const hmac = crypto.createHmac('sha256', secretKey);
  hmac.update(message);
  const hash = hmac.digest();
  return Buffer.from(hash).toString('base64');
};

const preHash = (timestamp, method, requestPath, body) => {
  return `${timestamp}${method.toUpperCase()}${requestPath}${body}`;
};

const parseParamsToStr = (params) => {
  const queryString = Object.entries(params)
    .map(([key, val]) => `${encodeURIComponent(key)}=${encodeURIComponent(val)}`)
    .join('&');
  return queryString ? `?${queryString}` : '';
};

const getBitgetWalletFutures = async () => {
  const timestamp = getTimestamp();
  const method = 'GET';
  const endpoint = '/api/v2/mix/account/accounts';
  const baseUrl = 'https://api.bitget.com';
  const queryParams = { productType: 'USDT-FUTURES' };

  const requestPath = endpoint + parseParamsToStr(queryParams);
  const signString = preHash(timestamp, method, requestPath, '');
  const signature = signMessage(signString, secretKey);

  const headers = {
    'ACCESS-KEY': apiKey,
    'ACCESS-SIGN': signature,
    'ACCESS-TIMESTAMP': timestamp.toString(),
    'ACCESS-PASSPHRASE': passphrase,
    'locale': 'en-US',
    'Content-Type': 'application/json',
  };

  try {
    const response = await axios.get(`${baseUrl}${requestPath}`, { headers });
    return response.data.data[0];
  } catch (error) {
    console.error('Error fetching Bitget wallet futures:', error.response ? error.response.data : error.message);
    throw error;
  }
};

const placeBitgetOrder = async (orderData) => {
    const timestamp = getTimestamp();
    const method = 'POST';
    const endpoint = '/api/v2/mix/order/place-order';
    const baseUrl = 'https://api.bitget.com';
  
    const body = JSON.stringify(orderData);
  
    const requestPath = endpoint;
    const signString = preHash(timestamp, method, requestPath, body);
    const signature = signMessage(signString, secretKey);
  
    const headers = {
      'ACCESS-KEY': apiKey,
      'ACCESS-SIGN': signature,
      'ACCESS-TIMESTAMP': timestamp.toString(),
      'ACCESS-PASSPHRASE': passphrase,
      'locale': 'en-US',
      'Content-Type': 'application/json',
    };
  
    try {
      const response = await axios.post(`${baseUrl}${requestPath}`, body, { headers });
      return response.data;
    } catch (error) {
      console.error('Error placing Bitget order:', error.response ? error.response.data : error.message);
    }
  };

app.post('/webhook', async (req, res) => {
  const webhookData = req.body;
  console.log('Webhook Data:', webhookData);

  try {
    const wallet = await getBitgetWalletFutures();
    console.log('Wallet Data:', wallet.available);
    const availableBalance = parseFloat(wallet.available);
    const orderSize_usdt = (availableBalance * 0.2).toFixed(6);
    const orderSize = orderSize_usdt / webhookData.price
    const orderData = {
        symbol: webhookData.symbol.replace('.P', ""),
        productType: 'USDT-FUTURES',
        marginMode: 'isolated',
        marginCoin: 'USDT',
        size: orderSize,
        side: webhookData.action.toLowerCase() === 'short' ? 'sell' : 'buy',
        orderType: 'market',
        presetStopSurplusPrice: webhookData.objective.toString(),
        presetStopLossPrice: webhookData.stop_loss.toString(),
        clientOid: Date.now().toString(),
        reduceOnly: 'NO',
        tradeSide: 'open'
      };
      console.log(orderData)
    const orderResponse = await placeBitgetOrder(orderData);
    res.status(200).json({ success: true, orderResponse });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
