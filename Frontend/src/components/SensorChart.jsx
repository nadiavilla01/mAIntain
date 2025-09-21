import React, { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
  CartesianGrid,
} from "recharts";

const COLORS = {
  power: "#facc15",
  speed: "#ef7d7d",
  temperature: "#38bdf8",
  vibration: "#22c55e",
};

export default function SensorChart({
  trend = {},
  availableSensors = [],
  rangePct = 100,
  smooth = false,
  mode = "compare", 
  selectedSensor = null,
}) {
  const { data, keys } = useMemo(() => {
    const keys = availableSensors.length
      ? availableSensors
      : Object.keys(trend || {});
    const maxLen = keys.reduce((m, k) => Math.max(m, (trend[k] || []).length), 0);
    const cut = Math.max(1, Math.floor((maxLen * rangePct) / 100));
    const points = new Array(cut).fill(0).map((_, i) => {
      const row = { t: `${i * 5}m` }; 
      keys.forEach((k) => {
        const arr = trend[k] || [];
        row[k] = typeof arr[i] === "number" ? arr[i] : null;
      });
      return row;
    });
    return { data: points, keys };
  }, [trend, availableSensors, rangePct]);

  const showKeys =
    mode === "single"
      ? selectedSensor
        ? [selectedSensor]
        : keys.slice(0, 1)
      : keys;

  return (
    <div style={{ width: "100%", height: 340 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 20, left: 8, bottom: 8 }}>
          <CartesianGrid stroke="#1b2944" strokeDasharray="3 3" />
          <XAxis
            dataKey="t"
            tick={{ fill: "#9fb3d9", fontSize: 12 }}
            stroke="#2a406a"
          />
          <YAxis tick={{ fill: "#9fb3d9", fontSize: 12 }} stroke="#2a406a" />
          <Tooltip
            contentStyle={{
              background: "#0b1426",
              border: "1px solid #274066",
              borderRadius: 10,
              color: "#e6eefb",
            }}
          />
          <Legend
            wrapperStyle={{ color: "#cfeaff", paddingTop: 8 }}
            iconType="plainline"
          />
          {showKeys.map((k) => (
            <Line
              key={k}
              type={smooth ? "monotone" : "linear"}
              dataKey={k}
              stroke={COLORS[k] || "#38bdf8"}
              strokeWidth={2.2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
