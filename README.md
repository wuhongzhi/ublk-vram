This is fork from https://github.com/theblazehen/vramblk.git, but use ublk instead of NBD.
about ublk, please see https://www.kernel.org/doc/html/latest/block/ublk.html

## Limitations
 
- Performance is limited by PCI-Express bandwidth, OpenCL overhead.
- Maximum size is limited by available OCL memory.
- Not recommended for critical data (no persistence).
- Requires root privileges for the server (`mlockall`, OpenCL).
- `mlockall` might fail if limits (`ulimit -l`) are too low or user lacks privileges.

---

## License

MIT

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
