#!/usr/bin/env python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args, unknown = parser.parse_known_args()
    print(f"Executed {__file__} with config={args.config} extra={unknown}")

if __name__ == '__main__':
    main()
