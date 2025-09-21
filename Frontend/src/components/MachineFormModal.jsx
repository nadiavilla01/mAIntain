import React, { useState, useEffect } from "react";
import {
  Modal,
  Box,
  Typography,
  TextField,
  Button,
  MenuItem,
} from "@mui/material";

const modalStyle = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: 420,
  p: 4,
  borderRadius: "16px",
  backdropFilter: "blur(16px)",
  WebkitBackdropFilter: "blur(16px)",
  background: "rgba(15, 23, 42, 0.65)",
  boxShadow: "0 0 20px rgba(56, 189, 248, 0.4)",
  border: "1px solid rgba(56, 189, 248, 0.3)",
  color: "#e2e8f0",
  display: "flex",
  flexDirection: "column",
  gap: 2,
};

const inputStyle = {
  "& label": { color: "#94a3b8" },
  "& .MuiOutlinedInput-root": {
    backgroundColor: "rgba(255, 255, 255, 0.05)",
    backdropFilter: "blur(2px)",
    borderRadius: "8px",
    "& fieldset": {
      borderColor: "#334155",
    },
    "&:hover fieldset": {
      borderColor: "#38bdf8",
    },
    "&.Mui-focused fieldset": {
      borderColor: "#38bdf8",
      boxShadow: "0 0 6px #38bdf8",
    },
  },
  "& input": {
    color: "#e2e8f0",
  },
  "& select": {
    color: "#e2e8f0",
  },
};

const MachineFormModal = ({ open, onClose, onSubmit, initialData }) => {
  const [form, setForm] = useState({
    name: "",
    location: "",
    status: "Normal",
    rul: "",
    lastUpdated: "Just now",
  });

  useEffect(() => {
    if (initialData) {
      setForm({
        ...initialData,
        lastUpdated: initialData.lastUpdated || "Just now",
      });
    }
  }, [initialData]);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = () => {
    onSubmit(form);
    onClose();
  };

  return (
    <Modal open={open} onClose={onClose}>
      <Box sx={modalStyle}>
        <Typography
          variant="h6"
          sx={{ color: "#38bdf8", fontWeight: "bold", mb: 1 }}
        >
          {initialData ? "Edit Machine" : "Add Machine"}
        </Typography>

        <TextField
          label="Name"
          name="name"
          value={form.name}
          onChange={handleChange}
          fullWidth
          sx={inputStyle}
        />

        <TextField
          label="Location"
          name="location"
          value={form.location}
          onChange={handleChange}
          fullWidth
          sx={inputStyle}
        />

        <TextField
          label="RUL"
          name="rul"
          value={form.rul}
          onChange={handleChange}
          fullWidth
          sx={inputStyle}
        />

        <TextField
          select
          label="Status"
          name="status"
          value={form.status}
          onChange={handleChange}
          fullWidth
          sx={inputStyle}
        >
          <MenuItem value="Normal">Normal</MenuItem>
          <MenuItem value="Unstable">Unstable</MenuItem>
          <MenuItem value="Critical">Critical</MenuItem>
        </TextField>

        <Button
          variant="contained"
          onClick={handleSubmit}
          fullWidth
          sx={{
            mt: 1,
            backgroundColor: "#0ea5e9",
            color: "#fff",
            fontWeight: "bold",
            borderRadius: "10px",
            boxShadow: "0 0 10px #38bdf8",
            "&:hover": {
              backgroundColor: "#0284c7",
              boxShadow: "0 0 14px #38bdf8",
            },
          }}
        >
          Save
        </Button>
      </Box>
    </Modal>
  );
};

export default MachineFormModal;
