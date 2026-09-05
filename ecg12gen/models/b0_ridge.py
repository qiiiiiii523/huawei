from pathlib import Path

import numpy as np


class StreamingRidge:
    """
    面向ECG逐采样点映射的Ridge模型。

    输入形状:
        [batch, input_leads, points]

    输出形状:
        [batch, output_leads, points]
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 12,
        alpha: float = 1.0,
        fit_intercept: bool = False,
    ):
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")

        if output_channels <= 0:
            raise ValueError("output_channels must be positive")

        if alpha < 0:
            raise ValueError("alpha must be non-negative")

        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)

        feature_count = self.input_channels + int(self.fit_intercept)

        # 使用float64累计，降低大量样本累加产生的数值误差
        self.xtx = np.zeros(
            (feature_count, feature_count),
            dtype=np.float64,
        )
        self.xty = np.zeros(
            (feature_count, self.output_channels),
            dtype=np.float64,
        )

        self.coef_ = None
        self.intercept_ = None
        self.n_samples_seen_ = 0

    def _validate_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray | None = None,
    ) -> None:
        if inputs.ndim != 3:
            raise ValueError(
                f"inputs must have shape [B, C, T], got {inputs.shape}"
            )

        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} input channels, "
                f"got {inputs.shape[1]}"
            )

        if not np.isfinite(inputs).all():
            raise ValueError("inputs contain NaN or Inf")

        if targets is None:
            return

        if targets.ndim != 3:
            raise ValueError(
                f"targets must have shape [B, 12, T], got {targets.shape}"
            )

        if targets.shape[0] != inputs.shape[0]:
            raise ValueError("input and target batch sizes do not match")

        if targets.shape[2] != inputs.shape[2]:
            raise ValueError("input and target point counts do not match")

        if targets.shape[1] != self.output_channels:
            raise ValueError(
                f"expected {self.output_channels} target channels, "
                f"got {targets.shape[1]}"
            )

        if not np.isfinite(targets).all():
            raise ValueError("targets contain NaN or Inf")

    def partial_fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> "StreamingRidge":
        """
        累计一个批次的充分统计量。
        """
        inputs = np.asarray(inputs)
        targets = np.asarray(targets)

        self._validate_batch(inputs, targets)

        # [B, C, T] -> [B*T, C]
        x = inputs.transpose(0, 2, 1).reshape(
            -1,
            self.input_channels,
        ).astype(np.float64, copy=False)

        # [B, O, T] -> [B*T, O]
        y = targets.transpose(0, 2, 1).reshape(
            -1,
            self.output_channels,
        ).astype(np.float64, copy=False)

        if self.fit_intercept:
            ones = np.ones((x.shape[0], 1), dtype=np.float64)
            x = np.concatenate((x, ones), axis=1)

        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.n_samples_seen_ += x.shape[0]

        return self

    def finalize(self) -> "StreamingRidge":
        """
        根据累计结果求解Ridge闭式解。
        """
        if self.n_samples_seen_ == 0:
            raise RuntimeError("no training samples have been accumulated")

        regularizer = np.eye(self.xtx.shape[0], dtype=np.float64)
        regularizer *= self.alpha

        # 截距项不进行Ridge惩罚
        if self.fit_intercept:
            regularizer[-1, -1] = 0.0

        weights = np.linalg.solve(
            self.xtx + regularizer,
            self.xty,
        )

        if self.fit_intercept:
            self.coef_ = weights[:-1].T
            self.intercept_ = weights[-1]
        else:
            self.coef_ = weights.T
            self.intercept_ = np.zeros(
                self.output_channels,
                dtype=np.float64,
            )

        return self

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        生成形状为[B, 12, T]的预测结果。
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("model has not been finalized or loaded")

        inputs = np.asarray(inputs)
        self._validate_batch(inputs)

        predictions = np.einsum(
            "oc,bct->bot",
            self.coef_,
            inputs,
            optimize=True,
        )

        predictions += self.intercept_[None, :, None]

        return predictions.astype(np.float32, copy=False)

    def save(self, path: str | Path) -> None:
        """
        保存模型系数。
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("cannot save an unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            coef=self.coef_,
            intercept=self.intercept_,
            alpha=np.asarray(self.alpha),
            fit_intercept=np.asarray(self.fit_intercept),
            input_channels=np.asarray(self.input_channels),
            output_channels=np.asarray(self.output_channels),
            n_samples_seen=np.asarray(self.n_samples_seen_),
        )

    @classmethod
    def load(cls, path: str | Path) -> "StreamingRidge":
        """
        从npz文件读取模型。
        """
        with np.load(path, allow_pickle=False) as data:
            model = cls(
                input_channels=int(data["input_channels"]),
                output_channels=int(data["output_channels"]),
                alpha=float(data["alpha"]),
                fit_intercept=bool(data["fit_intercept"]),
            )

            model.coef_ = data["coef"].astype(
                np.float64,
                copy=False,
            )
            model.intercept_ = data["intercept"].astype(
                np.float64,
                copy=False,
            )
            model.n_samples_seen_ = int(data["n_samples_seen"])

        return model