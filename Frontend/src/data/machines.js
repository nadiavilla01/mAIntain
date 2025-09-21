const machines = [
    {
      id: "1",
      name: "Stamping Machine #1",
      status: "Running",
      operatingTime: "56h 22m",
      output: "470",
      temp: "85°c",
      tempTrend: [20, 32, 40, 48, 66, 75],
      alerts: ["High temperature detected"],
      sensors: {
        speed: "1450 rpm",
        vibration: "4.2 mm/s",
        power: "1.8 kW"
      }
    },
    // Add more machines here
  ];
  
  export default machines;
  