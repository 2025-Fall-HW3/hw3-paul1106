"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import argparse
import warnings
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

start = "2019-01-01"
end = "2024-04-01"

# Initialize df and df_returns
df = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start=start, end=end, auto_adjust = False)
    df[asset] = raw['Adj Close']

df_returns = df.pct_change().fillna(0)


"""
Problem 1: 

Implement an equal weighting strategy as dataframe "eqw". Please do "not" include SPY.
"""


class EqualWeightPortfolio:
    def __init__(self, exclude):
        self.exclude = exclude

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        # # ===========================================
        # print("\n" + "="*30)
        # print("【DEBUG: 檢查資料結構】")
        # print(f"1. 資料維度 (Shape): {df_returns.shape}")
        # print("   (解釋: Rows 是天數, Columns 是資產數量)")
        # print("\n2. 前 5 筆報酬率資料 (df_returns.head()):")
        # print(df_returns.head())
        # print("\n3. 我們要分配權重的資產列表 (assets):")
        # print(assets)
        # print("="*30 + "\n")
        # # ===========================================
        # """
        # TODO: Complete Task 1 Below
        n = len(assets)
        # TODO: Complete Task 1 Above
        self.portfolio_weights[assets] = 1.0 / n
        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Problem 2:

Implement a risk parity strategy as dataframe "rp". Please do "not" include SPY.
"""


class RiskParityPortfolio:
    def __init__(self, exclude, lookback=50):
        self.exclude = exclude
        self.lookback = lookback

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        """
        TODO: Complete Task 2 Below
        """
        rolling_std = df_returns[assets].iloc[1:].rolling(window=self.lookback, min_periods=self.lookback).std()
        inv_vol = 1.0 / rolling_std
        inv_vol.replace([np.inf, -np.inf], 0.0, inplace=True)
        sum_inv_vol = inv_vol.sum(axis=1)
        weights = inv_vol.div(sum_inv_vol, axis=0)
        self.portfolio_weights[assets] = weights.shift(1)

        """
        TODO: Complete Task 2 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)
        print(self.portfolio_weights.head(55))
        # print(self.portfolio_weights.tail())

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Problem 3:

Implement a Markowitz strategy as dataframe "mv". Please do "not" include SPY.
"""


class MeanVariancePortfolio:
    def __init__(self, exclude, lookback=50, gamma=0):
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = df.columns[df.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(index=df.index, columns=df.columns)

        for i in range(self.lookback + 1, len(df)):
            R_n = df_returns.copy()[assets].iloc[i - self.lookback : i]
            self.portfolio_weights.loc[df.index[i], assets] = self.mv_opt(
                R_n, self.gamma
            )

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def mv_opt(self, R_n, gamma):
        Sigma = R_n.cov().values
        mu = R_n.mean().values
        n = len(R_n.columns)

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("DualReductions", 0)
            env.start()
            with gp.Model(env=env, name="portfolio") as model:
                """
                TODO: Complete Task 3 Below
                """

                # --- Step 1: 定義決策變數 (Decision Variables) ---
                # w 是權重向量，長度為 n
                # lb=0.0 代表 w >= 0 (Long-only constraint)
                # ub=1.0 代表 w <= 1 (其實 sum=1 已經隱含了，但加了也無妨)
                w = model.addMVar(n, name="w", lb=0.0, ub=1.0)

                # --- Step 2: 定義目標函數 (Objective Function) ---
                # 公式: Maximize ( w.T * mu - (gamma/2) * w.T * Sigma * w )
                
                # 預期報酬 (Portfolio Return)
                port_return = w @ mu
                
                # 投資組合變異數 (Portfolio Variance)
                # 注意：因為 w 是 MVar，這裡可以直接用 @ 做矩陣乘法
                port_risk = w @ Sigma @ w
                
                # 設定最大化目標
                # Gurobi 會自動識別這是一個 QP (Quadratic Programming) 問題
                model.setObjective(port_return - (gamma / 2) * port_risk, gp.GRB.MAXIMIZE)

                # --- Step 3: 設定限制條件 (Constraints) ---
                # 預算限制：所有權重加總必須等於 1
                model.addConstr(w.sum() == 1, name="budget")

                """
                TODO: Complete Task 3 Above
                """
                
                # --- Step 4: 求解 (Optimize) ---
                model.optimize()

                # --- Step 5: 處理求解結果與例外狀況 ---
                # 檢查是否無解 (Infeasible) 或 無界 (Unbounded)
                if model.status == gp.GRB.INF_OR_UNBD:
                    print("Model status is INF_OR_UNBD. Reoptimizing with DualReductions set to 0.")
                    model.setParam("DualReductions", 0)
                    model.optimize()
                    
                if model.status == gp.GRB.INFEASIBLE:
                    print("Model is infeasible.")
                    # 如果無解，回傳平均分配 (當作 fallback)
                    return np.ones(n) / n
                    
                elif model.status == gp.GRB.UNBOUNDED:
                    print("Model is unbounded.")
                    return np.ones(n) / n

                # 如果成功找到最佳解 (Optimal)
                if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                    # 提取 w 的數值 (.X 屬性)
                    return w.X
                
                # 如果發生其他錯誤，回傳平均分配避免程式崩潰
                return np.ones(n) / n

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = df_returns.copy()
        assets = df.columns[df.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader import AssignmentJudge

    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 1"
    )
    """
    NOTE: For Assignment Judge
    """
    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader.py
    judge.run_grading(args)
