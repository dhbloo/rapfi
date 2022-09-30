Module["sendCommand"] = Module["sendCommand"] || null;
Module["receiveStdout"] = Module["receiveStdout"] || ((o) => console.log(o));
Module["receiveStderr"] = Module["receiveStderr"] || ((o) => console.error(o));
Module["onEngineReady"] = Module["onEngineReady"] || (() => {});

if (!Module["preRun"]) Module["preRun"] = [];
Module["preRun"].push(function () {
  let stdin_buffer = { buffer: "", index: 0 };
  let stdout_buffer = "";
  let stderr_buffer = "";

  function stdin() {
    if (stdin_buffer.index < stdin_buffer.buffer.length)
      return stdin_buffer.buffer.charCodeAt(stdin_buffer.index++);
    else return null;
  }

  function stdout(char) {
    if (!char || char == "\n".charCodeAt(0)) {
      Module["receiveStdout"](stdout_buffer);
      stdout_buffer = "";
    } else stdout_buffer += String.fromCharCode(char);
  }

  function stderr(char) {
    if (!char || char == "\n".charCodeAt(0)) {
      Module["receiveStderr"](stderr_buffer);
      stderr_buffer = "";
    } else stderr_buffer += String.fromCharCode(char);
  }

  // Redirect stdin, stdout, stderr
  FS.init(stdin, stdout, stderr);
  let execute_command = Module["cwrap"]("gomocupLoopOnce", "number", []);
  Module["sendCommand"] = function (data) {
    stdin_buffer.buffer = data + "\n";
    stdin_buffer.index = 0;
    execute_command();
  };
});
Module["onRuntimeInitialized"] = function () {
  Module["onEngineReady"]();
};
