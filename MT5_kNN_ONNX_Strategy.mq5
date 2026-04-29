#property strict
#include <Trade/Trade.mqh>

CTrade trade;

// === INPUTS ===
input double InpLot = 0.1;
input double InpEntryProbThreshold = 0.6;
input double InpMinProbGap = 0.05;

// === EMBED MODEL ===
#resource "ml_strategy_classifier_knn.onnx" as uchar ModelData[]

long model;

vectorf input_data;
vectorf output_data;

const long input_shape[] = {1,10};
const long output_shape[] = {1,3};

double features[10];

int OnInit()
{
   model = OnnxCreateFromBuffer(ModelData, ONNX_DEFAULT);

   if(model == INVALID_HANDLE)
   {
      Print("ONNX load failed ", GetLastError());
      return INIT_FAILED;
   }

   OnnxSetInputShape(model, 0, input_shape);
   OnnxSetOutputShape(model, 0, output_shape);

   input_data.Resize(10);
   output_data.Resize(3);

   return INIT_SUCCEEDED;
}


void BuildFeatures()
{
   double c0 = iClose(_Symbol, PERIOD_M15, 0);
   double c1 = iClose(_Symbol, PERIOD_M15, 1);
   double c3 = iClose(_Symbol, PERIOD_M15, 3);
   double c5 = iClose(_Symbol, PERIOD_M15, 5);
   double c10 = iClose(_Symbol, PERIOD_M15, 10);

   features[0]=c0/c1-1;
   features[1]=c0/c3-1;
   features[2]=c0/c5-1;
   features[3]=c0/c10-1;

   for(int i=4;i<10;i++) features[i]=features[0];
}


void OnTick()
{
   if(Bars(_Symbol, PERIOD_M15)<50) return;

   BuildFeatures();

   for(int i=0;i<10;i++) input_data[i]=(float)features[i];

   if(!OnnxRun(model, ONNX_NO_CONVERSION, input_data, output_data))
      return;

   double p_sell=output_data[0];
   double p_flat=output_data[1];
   double p_buy=output_data[2];

   double best = MathMax(p_buy,p_sell);
   double second = MathMax(p_flat, MathMin(p_buy,p_sell));
   double gap = best-second;

   if(best < InpEntryProbThreshold) return;
   if(gap < InpMinProbGap) return;

   if(PositionSelect(_Symbol)) return;

   if(p_buy>p_sell)
      trade.Buy(InpLot);
   else
      trade.Sell(InpLot);
}

double OnTester() {
  double profit = TesterStatistics(STAT_PROFIT);
  double pf = TesterStatistics(STAT_PROFIT_FACTOR);
  double recovery = TesterStatistics(STAT_RECOVERY_FACTOR);
  double dd_percent = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
  double trades = TesterStatistics(STAT_TRADES);

  // Penalty if there are too few transactions
  double trade_penalty = 1.0;
  if (trades < 20)
    trade_penalty = 0.25;
  else if (trades < 50)
    trade_penalty = 0.60;

  // Robust score, not only brut profit
  double score = 0.0;

  if (dd_percent >= 0.0)
    score =
        (profit * MathMax(pf, 0.01) * MathMax(recovery, 0.01) * trade_penalty) /
        (1.0 + dd_percent);

  return score;
}
