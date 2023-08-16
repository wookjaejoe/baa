from __future__ import annotations

import numpy as np
import pandas as pd
import quantstats
import yfinance as yf

from asset import Assets

MOMENTUM_WEIGHTS = {period_month: 12 / period_month for period_month in [1, 3, 6, 12]}


def load_day_chart() -> pd.DataFrame:
    try:
        return pd.read_csv("chart.csv", parse_dates=["Date"]).set_index("Date")
    except:
        data = yf.download(Assets.all(), period="max")
        data = data[[col for col in data.columns if col[0] == "Adj Close"]]
        data.columns = [col[1] for col in data.columns]
        data.to_csv("chart.csv")
        return data


def main():
    # 일봉 차트 로드
    chart = load_day_chart().dropna()

    # 월봉 리샘플링
    month_chart = chart.resample("M").last()

    def momentum(row: pd.Series, months: int) -> pd.Series:
        pos = month_chart[row.index].index.get_loc(row.name)
        assert pos - months >= 0, "Not enough data."
        return row / month_chart[row.index].iloc[pos - months] - 1

    def momentum_score(row):
        # 기간별 모멘텀의 가중합을 구하여 반환
        try:
            return sum([momentum(row, m) * w for m, w in MOMENTUM_WEIGHTS.items()])
        except AssertionError:
            return pd.Series(index=row.index, name=row.name, data=np.nan)

    # 모멘텀 스코어 계산 - 1, 3, 6, 12 수익률 가중합
    mmt = month_chart[list(set(Assets.canaries + Assets.offensives))].apply(momentum_score, axis=1).dropna()

    # 모멘텀 스코어로 카나리 시그널 발생 시점 산출
    canary_signal = mmt[Assets.canaries].le(0).any(axis=1)

    # 일별 모든 자산 수익률 계산
    result = chart.rolling(2).apply(lambda row: row[1] / row[0] - 1)

    def select_defensives(row: pd.Series):
        pos = month_chart.index.get_loc(row.name)
        current = row[Assets.defensives]
        ma = month_chart[Assets.defensives].iloc[pos - 11:pos + 1].mean()  # 최근 12개월 이동평균
        return ["BIL" if value < 1 else name for name, value in (current / ma).nlargest(3).items()]

    # 카나리 시그널에 기반한 일별 포트폴리오 구성
    result["port"] = pd.concat([
        month_chart.reindex(canary_signal.index)[canary_signal].apply(select_defensives, axis=1),
        mmt[~canary_signal][Assets.offensives].apply(lambda row: row.nlargest(1).index.tolist(), axis=1)
    ]).sort_index().fillna(np.nan).reindex(chart.index, method="ffill").shift(1).dropna()

    # 포트폴리오 수익률
    result["port_profits"] = result.apply(
        lambda row: np.nan if row["port"] is np.nan else row[row["port"]].mean(),
        axis=1
    )

    # 포트폴리오 누적 수익률
    result["balance"] = (result["port_profits"] + 1).cumprod()
    dates = result.dropna().index
    years = (dates.max() - dates.min()).days / 365

    # 장중 MDD는 무시
    balance = result["balance"]
    cummax = balance.cummax()
    cummax_index = cummax.to_frame("cummax").groupby("cummax").apply(lambda x: x.index.min())
    dd = (balance - cummax) / cummax
    mdd = dd.cummin()
    result = pd.concat([
        result,
        cummax.to_frame("cummax"),
        pd.merge(cummax.to_frame("cummax"), cummax_index.to_frame("cummax_pos"), left_on="cummax", right_index=True),
        dd.to_frame("dd"),
        mdd.to_frame("mdd"),
    ], axis=1)

    quantstats.reports.html(result["port_profits"], "SPY", output="report.html")

    cagr = result["balance"][-1] ** (1 / years) - 1
    print(cagr, mdd[-1])


if __name__ == '__main__':
    main()
