Module["sendCommand"] = Module["sendCommand"] || null;
Module["terminate"] = Module["terminate"] || null;
Module["onReceiveStdout"] = Module["onReceiveStdout"] || ((o) => console.log(o));
Module["onReceiveStderr"] = Module["onReceiveStderr"] || ((o) => console.error(o));
Module["onExit"] = Module["onExit"] || ((code) => console.log("exited with code " + code));
Module["noExitRuntime"] = true; // Only exit when we explicitly want to do so

if (!Module["preRun"]) Module["preRun"] = [];
Module["preRun"].push(function () {
  let stdin_queue = [];
  let stdin_buffer = { data: "", index: 0 };
  let stdout_buffer = "";
  let stderr_buffer = "";

  function stdin() {
    if (stdin_buffer.index < stdin_buffer.data.length)
      return stdin_buffer.data.charCodeAt(stdin_buffer.index++);
    else if (stdin_queue.length > 0) {
      stdin_buffer.data = stdin_queue.shift();
      stdin_buffer.index = 0;
      return stdin_buffer.data.charCodeAt(stdin_buffer.index++);
    } else return null;
  }

  const newline_charcode = "\n".charCodeAt(0)
  const stdout_fn = Module["onReceiveStdout"]
  const stderr_fn = Module["onReceiveStderr"]

  function stdout(char) {
    if (!char || char == newline_charcode) {
      stdout_fn(stdout_buffer);
      stdout_buffer = "";
    } else stdout_buffer += String.fromCharCode(char);
  }

  function stderr(char) {
    if (!char || char == newline_charcode) {
      stderr_fn(stderr_buffer);
      stderr_buffer = "";
    } else stderr_buffer += String.fromCharCode(char);
  }

  // Redirect stdin, stdout, stderr
  FS.init(stdin, stdout, stderr);
  const execute_command = Module["cwrap"]("gomocupLoopOnce", "number", []);
  Module["sendCommand"] = function (data) {
    stdin_queue.push(data + "\n");
    execute_command();
  };
  Module["terminate"] = function () {
    Module["_emscripten_force_exit"](0);
  };
});
