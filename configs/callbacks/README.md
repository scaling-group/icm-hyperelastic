# Callback Configuration

This directory contains the configuration files for the callbacks. There are two types of callback configuration files:

- `single_callback_name.yaml`: Configuration for a single callback.
- `many_callbacks_project_name.yaml`: listing the configuration files of individual callbacks to be used in the project.

For your project, you should create a new `many_callbacks_project_name.yaml` file to list the configuration files of individual callbacks. If you need to implement a new callback, you can create a corresponding `single_callback_name.yaml` file, and add it to the `many_callbacks_project_name.yaml` file.
