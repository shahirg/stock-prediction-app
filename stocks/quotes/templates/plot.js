var trace1 = {
  type: "scatter",
  mode: "lines",
  name: "AAPL High",
  x: unpack(rows, "Date"),
  y: unpack(rows, "AAPL.High"),
  line: { color: "#17BECF" },
};

var trace2 = {
  type: "scatter",
  mode: "lines",
  name: "AAPL Low",
  x: unpack(rows, "Date"),
  y: unpack(rows, "AAPL.Low"),
  line: { color: "#7F7F7F" },
};

var data = [trace1, trace2];

var layout = {
  title: "Basic Time Series",
};

Plotly.newPlot("myDiv", data, layout);
