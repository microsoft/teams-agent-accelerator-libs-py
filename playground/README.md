# Memory Playground UI

A simple web application to easily view and interact with the memory module.

## Setup Instructions

To set up and run the playground, follow these steps:

1. Navigate to this folder in the termainl.
1. Set up the dependencies by doing `pnpm install` in the root folder.
1. Run `pnpm run dev` to start the application on port 5173.
1. Ensure that the memory server is configured and running locally. See `packages/memory-server`.

## Notes
The memory server base url is defined in the `api/memoryServer.ts` file. By default it points to http://127.0.0.1:8000, i.e localhost:8000

# Disclaimer

Some components were taken from [TailAdmin](https://tailadmin.com/)'s free to use component package, which is under the MIT License.