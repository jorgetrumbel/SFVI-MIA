import { useState } from "react";

function TaskForm({ createTask }) {
  const [titulo, setTitulo] = useState("");
  const [des, setDes] = useState("");

  const haddleSubmit = (e) => {
    e.preventDefault();
    createTask({
      title: titulo,
      description: des
    });
    setTitulo('');
    setDes('');
  };

  return (
    <div>
      <form onSubmit={haddleSubmit}>
        <input
          type="text"
          placeholder="Escribe tu tarea"
          onChange={(e) => {
            setTitulo(e.target.value);
          }}
          value={titulo}
          autoFocus
        />
        <textarea
          placeholder="Ingrese la descripcion"
          onChange={(e) => {
            setDes(e.target.value);
          }}
          value={des}
        ></textarea>
        <button>Guardar</button>
      </form>
    </div>
  );
}

export default TaskForm;
