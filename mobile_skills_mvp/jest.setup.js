// Jest setup file
global.__DEV__ = true;

// Mock console methods to reduce noise in tests
global.console = {
  ...console,
  error: jest.fn(),
  log: jest.fn(),
  warn: jest.fn(),
};
