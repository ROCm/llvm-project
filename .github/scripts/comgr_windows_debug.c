// Diagnose a native Comgr test without changing the test or DLL.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <dbghelp.h>
#include <psapi.h>
#include <stdio.h>

static void print_module(HANDLE Process, HANDLE File, void *Base) {
  wchar_t Path[32768];
  DWORD Length = File ? GetFinalPathNameByHandleW(File, Path, 32768, 0) : 0;
  if (!Length || Length >= 32768)
    Length = GetMappedFileNameW(Process, Base, Path, 32768);
  if (Length && Length < 32768)
    printf("module %p %ls\n", Base, Path);
  else
    printf("module %p (path unavailable: %lu)\n", Base, GetLastError());
}

static int record_exception(HANDLE Process, DWORD ProcessId, DWORD ThreadId,
                            EXCEPTION_RECORD *Record, const wchar_t *DumpPath) {
  HANDLE Thread = OpenThread(THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                             FALSE, ThreadId);
  CONTEXT Context = {0};
  Context.ContextFlags = CONTEXT_FULL;
  if (!Thread || !GetThreadContext(Thread, &Context)) {
    fprintf(stderr, "GetThreadContext failed: %lu\n", GetLastError());
    if (Thread)
      CloseHandle(Thread);
    return 1;
  }

  EXCEPTION_POINTERS Pointers = {Record, &Context};
  MINIDUMP_EXCEPTION_INFORMATION Exception = {ThreadId, &Pointers, FALSE};
  HANDLE Dump = CreateFileW(DumpPath, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL, NULL);
  int Failed = 0;
  if (Dump == INVALID_HANDLE_VALUE ||
      !MiniDumpWriteDump(Process, ProcessId, Dump,
                         MiniDumpWithThreadInfo | MiniDumpWithUnloadedModules |
                             MiniDumpWithIndirectlyReferencedMemory,
                         &Exception, NULL, NULL)) {
    fprintf(stderr, "Writing crash dump failed: %lu\n", GetLastError());
    Failed = 1;
  } else {
    printf("crash dump: %ls\n", DumpPath);
  }
  if (Dump != INVALID_HANDLE_VALUE)
    CloseHandle(Dump);

  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
  if (SymInitialize(Process, NULL, TRUE)) {
    STACKFRAME64 Frame = {0};
    Frame.AddrPC.Offset = Context.Rip;
    Frame.AddrFrame.Offset = Context.Rbp;
    Frame.AddrStack.Offset = Context.Rsp;
    Frame.AddrPC.Mode = Frame.AddrFrame.Mode = Frame.AddrStack.Mode =
        AddrModeFlat;
    for (unsigned I = 0; I < 64; ++I) {
      if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, Process, Thread, &Frame,
                       &Context, NULL, SymFunctionTableAccess64,
                       SymGetModuleBase64, NULL) ||
          !Frame.AddrPC.Offset)
        break;
      union {
        SYMBOL_INFO Info;
        char Storage[sizeof(SYMBOL_INFO) + MAX_SYM_NAME];
      } Symbol = {0};
      Symbol.Info.SizeOfStruct = sizeof(SYMBOL_INFO);
      Symbol.Info.MaxNameLen = MAX_SYM_NAME;
      DWORD64 Displacement = 0;
      DWORD64 Base = SymGetModuleBase64(Process, Frame.AddrPC.Offset);
      printf("frame %u: 0x%llx module+0x%llx", I,
             (unsigned long long)Frame.AddrPC.Offset,
             (unsigned long long)(Frame.AddrPC.Offset - Base));
      if (SymFromAddr(Process, Frame.AddrPC.Offset, &Displacement,
                      &Symbol.Info))
        printf(" %s+0x%llx", Symbol.Info.Name,
               (unsigned long long)Displacement);
      putchar('\n');
    }
    SymCleanup(Process);
  } else {
    fprintf(stderr, "SymInitialize failed: %lu\n", GetLastError());
  }
  CloseHandle(Thread);
  return Failed;
}

int wmain(int Argc, wchar_t **Argv) {
  if (Argc != 3) {
    fprintf(stderr, "usage: comgr-windows-debug.exe TEST.exe CRASH.dmp\n");
    return 2;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  STARTUPINFOW Startup = {0};
  Startup.cb = sizeof(Startup);
  PROCESS_INFORMATION Process = {0};
  if (!CreateProcessW(Argv[1], NULL, NULL, NULL, FALSE, DEBUG_ONLY_THIS_PROCESS,
                      NULL, NULL, &Startup, &Process)) {
    fprintf(stderr, "CreateProcess failed: %lu\n", GetLastError());
    return 2;
  }
  CloseHandle(Process.hThread);
  BOOL InitialBreakpoint = TRUE;
  int Failed = 0;
  for (;;) {
    DEBUG_EVENT Event;
    if (!WaitForDebugEvent(&Event, INFINITE)) {
      fprintf(stderr, "WaitForDebugEvent failed: %lu\n", GetLastError());
      CloseHandle(Process.hProcess);
      return 2;
    }
    DWORD ContinueStatus = DBG_CONTINUE;
    switch (Event.dwDebugEventCode) {
    case CREATE_PROCESS_DEBUG_EVENT:
      print_module(Process.hProcess, Event.u.CreateProcessInfo.hFile,
                   Event.u.CreateProcessInfo.lpBaseOfImage);
      CloseHandle(Event.u.CreateProcessInfo.hFile);
      break;
    case LOAD_DLL_DEBUG_EVENT:
      print_module(Process.hProcess, Event.u.LoadDll.hFile,
                   Event.u.LoadDll.lpBaseOfDll);
      if (Event.u.LoadDll.hFile) {
        CloseHandle(Event.u.LoadDll.hFile);
      }
      break;
    case EXCEPTION_DEBUG_EVENT: {
      EXCEPTION_RECORD *Record = &Event.u.Exception.ExceptionRecord;
      printf("exception 0x%08lx at %p, first chance=%lu\n",
             Record->ExceptionCode, Record->ExceptionAddress,
             Event.u.Exception.dwFirstChance);
      for (DWORD I = 0; I < Record->NumberParameters; ++I)
        printf("  parameter %lu: 0x%llx\n", I,
               (unsigned long long)Record->ExceptionInformation[I]);
      if (InitialBreakpoint && Record->ExceptionCode == EXCEPTION_BREAKPOINT) {
        InitialBreakpoint = FALSE;
      } else {
        ContinueStatus = DBG_EXCEPTION_NOT_HANDLED;
        if (!Event.u.Exception.dwFirstChance)
          Failed |= record_exception(Process.hProcess, Process.dwProcessId,
                                     Event.dwThreadId, Record, Argv[2]);
      }
      break;
    }
    case EXIT_PROCESS_DEBUG_EVENT:
      printf("process exit: 0x%08lx\n", Event.u.ExitProcess.dwExitCode);
      Failed |= Event.u.ExitProcess.dwExitCode != 0;
      ContinueDebugEvent(Event.dwProcessId, Event.dwThreadId, DBG_CONTINUE);
      CloseHandle(Process.hProcess);
      return Failed;
    }
    if (!ContinueDebugEvent(Event.dwProcessId, Event.dwThreadId,
                            ContinueStatus)) {
      fprintf(stderr, "ContinueDebugEvent failed: %lu\n", GetLastError());
      CloseHandle(Process.hProcess);
      return 2;
    }
  }
}
