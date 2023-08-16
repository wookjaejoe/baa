class Assets:
    # offensives = ["QQQ", "VWO", "VEA", "BND"]
    offensives = ["QQQ", "EFA", "EEM", "AGG"]
    defensives = ["TIP", "DBC", "BIL", "IEF", "TLT", "LQD", "BND"]
    canaries = ["SPY", "VEA", "VWO", "BND"]

    @staticmethod
    def all() -> set[str]:
        return (
            set(Assets.offensives)
            | set(Assets.defensives)
            | set(Assets.canaries)
        )

    @staticmethod
    def is_offensives(asset: str):
        return asset in Assets.offensives

    @staticmethod
    def is_defensives(asset: str):
        return asset in Assets.defensives

    @staticmethod
    def is_protectives(asset: str):
        return asset in Assets.canaries
