import { createContext } from "react";

export const TaskContext = createContext();

function TaskContextProvider(props) {
  let x = 20;
  return (
    <TaskContext.Provider value={x}>{props.children}</TaskContext.Provider>
  );
}
