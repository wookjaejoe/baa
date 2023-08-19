from __future__ import annotations

import numpy as np
import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt

from asset import Assets

MOMENTUM_WEIGHTS = {period_month: 12 / period_month for period_month in [1, 3, 6, 12]}


def main():
    # 일봉 차트 로드
    chart = yf.download(Assets.all(), period="max")
    chart = chart[[col for col in chart.columns if col[0] == "Adj Close"]]
    chart.columns = [col[1] for col in chart.columns]
    chart = chart.dropna()

    # 월봉으로 리샘플링
    month_chart = chart.resample("M").last()

    def rate_of_return(row: pd.Series, months: int) -> pd.Series:
        """
        특정 시점 n개월 수익률 계산
        :param row: 특정 시점 주가
        :param months: 개월 수
        :return: 자산별 수익률
        """
        pos = month_chart[row.index].index.get_loc(row.name)
        assert pos - months >= 0, "Not enough data."
        return row / month_chart[row.index].iloc[pos - months] - 1

    def momentum_score(row: pd.Series):
        """
        특정 시점 모멘텀 스코어 계산
        :param row: 특정 시점 주가
        :return: 자산별 모멘텀 스코어
        """
        try:
            return sum([rate_of_return(row, m) * w for m, w in MOMENTUM_WEIGHTS.items()])
        except AssertionError:
            return pd.Series(index=row.index, name=row.name, data=np.nan)

    # 모멘텀 스코어 계산
    mmt = month_chart[list(set(Assets.canaries + Assets.offensives))].apply(momentum_score, axis=1).dropna()

    # 카나리 시그널 계산
    canary_signal = mmt[Assets.canaries].le(0).any(axis=1)

    def select_defensives(row: pd.Series):
        """
        방어 자산 중 12개월 수익률이 가장 높은 3개 자산 반환
        :param row: 특정 시점 주가
        :return: 선택된 3가지 방어 자산
        """
        pos = month_chart.index.get_loc(row.name)
        current = row[Assets.defensives]
        ma = month_chart[Assets.defensives].iloc[pos - 11:pos + 1].mean()  # 최근 12개월 이동평균
        return ["BIL" if value < 1 else name for name, value in (current / ma).nlargest(3).items()]

    # 카나리 시그널에 기반한 일별 포트폴리오 구성
    port = pd.DataFrame()
    port["tickers"] = pd.concat([
        month_chart.reindex(canary_signal.index)[canary_signal].apply(select_defensives, axis=1),
        mmt[~canary_signal][Assets.offensives].apply(lambda row: row.nlargest(1).index.tolist(), axis=1)
    ]).sort_index().fillna(np.nan).reindex(chart.index, method="ffill").shift(1).dropna()

    # 자산별 일별 수익률 계산
    returns = chart.rolling(2).apply(lambda row: row[1] / row[0] - 1)
    port = pd.concat([port, returns], axis=1)

    # 포트폴리오 수익률 계산
    port["return"] = port.dropna().apply(lambda row: row[row["tickers"]].mean(), axis=1)
    port = port[["tickers", "return"]].dropna()

    # 평가금액(최초금액 1로 가정)
    port["balance"] = (port["return"] + 1).cumprod()

    # 기간별 전략 성능 관련 상세 정보
    dates = port.dropna().index
    years = (dates.max() - dates.min()).days / 365
    cummax = port["balance"].cummax()
    cummax_pos = cummax.to_frame("cummax").groupby("cummax").apply(lambda x: x.index.min())
    dd = (port["balance"] - cummax) / cummax
    mdd = dd.cummin()
    cagr = port["balance"][-1] ** (1 / years) - 1
    cum_return = ((port["balance"] - 1) * 100)
    result = pd.concat([
        port,
        pd.merge(cummax.to_frame("cummax"), cummax_pos.to_frame("cummax_pos"), left_on="cummax", right_index=True),
        dd.to_frame("dd"),
        mdd.to_frame("mdd"),
        cum_return.to_frame("cum_return")
    ], axis=1)

    # 결과 요약
    summary = pd.Series({
        "Start date": result.index.min(),
        "End date": result.index.max(),
        "Cumulative Return(%)": result["cum_return"][-1],
        "CAGR(%)": cagr * 100,
        "MDD(%)": result["mdd"][-1] * 100,
        "Sharpe": result["return"].mean() / result["return"].std(),
    })
    print(summary)

    # 누적 수익률 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(result.index, cum_return, label="Strategy")
    for ticker in Assets.all():
        plt.plot(
            result.index,
            ((returns[ticker].reindex(result.index) + 1).cumprod() - 1) * 100,
            label=ticker,
            alpha=0.5
        )

    plt.title('Cumulative Return(%)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return(%)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
