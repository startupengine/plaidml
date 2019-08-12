def _tpl(ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    ctx.template(
        out,
        Label("@com_intel_plaidml//vendor/cm:%s.tpl" % tpl),
        substitutions,
    )

def _execute(ctx, cmdline, error_msg = None, error_details = None, empty_stdout_fine = False):
    """Executes an arbitrary shell command.

    Args:
        ctx: The repository context.
        cmdline: list of strings, the command to execute
        error_msg: string, a summary of the error if the command fails
        error_details: string, details about the error or steps to fix it
        empty_stdout_fine: bool, if True, an empty stdout result is fine, otherwise
        it's an error
    Return:
        the result of ctx.execute(cmdline)
    """
    result = ctx.execute(cmdline)
    if result.stderr or not (empty_stdout_fine or result.stdout):
        fail(
            "\n".join([
                error_msg.strip() if error_msg else "Repository command failed",
                result.stderr.strip(),
                error_details if error_details else "",
            ]),
        )
    return result

def _read_dir(ctx, src_dir):
    """Returns a string with all files in a directory.

    Finds all files inside a directory, traversing subfolders and following
    symlinks. The returned string contains the full path of all files
    separated by line breaks.
    """

    find_result = _execute(
        ctx,
        ["find", src_dir, "-follow", "-type", "f"],
        empty_stdout_fine = True,
    )
    result = find_result.stdout
    return result

def _norm_path(path):
    """Returns a path with '/' and remove the trailing slash."""
    path = path.replace("\\", "/")
    if path[-1] == "/":
        path = path[:-1]
    return path

def symlink_genrule_for_dir(ctx, src_dir, dest_dir, genrule_name, src_files = [], dest_files = []):
    """Returns a genrule to symlink(or copy if on Windows) a set of files.

    If src_dir is passed, files will be read from the given directory; otherwise
    we assume files are in src_files and dest_files
    """
    if src_dir != None:
        src_dir = _norm_path(src_dir)
        dest_dir = _norm_path(dest_dir)
        files = "\n".join(sorted(_read_dir(ctx, src_dir).splitlines()))

        # Create a list with the src_dir stripped to use for outputs.
        dest_files = files.replace(src_dir, "").splitlines()
        src_files = files.splitlines()
    command = []

    # We clear folders that might have been generated previously to avoid undesired inclusions
    if genrule_name == "cuda-include":
        command.append('if [ -d "$(@D)/cm/include" ]; then rm -rf $(@D)/cm/include; fi')
    elif genrule_name == "cuda-lib":
        command.append('if [ -d "$(@D)/cm/lib" ]; then rm -rf $(@D)/cm/lib; fi')
    outs = []
    for i in range(len(dest_files)):
        if dest_files[i] != "":
            # If we have only one file to link we do not want to use the dest_dir, as
            # $(@D) will include the full path to the file.
            dest = "$(@D)/" + dest_dir + dest_files[i] if len(dest_files) != 1 else "$(@D)/" + dest_files[i]
            command.append("mkdir -p $$(dirname {})".format(dest))
            command.append('ln -s "{}" "{}"'.format(src_files[i], dest))
            outs.append('        "{}{}",'.format(dest_dir, dest_files[i]))
    return _genrule(src_dir, genrule_name, command, outs)

def _genrule(src_dir, genrule_name, command, outs):
    """Returns a string with a genrule.

    Genrule executes the given command and produces the given outputs.
    """
    return "\n".join([
        "genrule(",
        '    name = "{}",'.format(genrule_name),
        "    outs = [",
    ] + outs + [
        "    ],",
        '    cmd = """',
    ] + command + [
        '    """,',
        ")",
    ])

def _create_cm_dummy_repository(ctx):
    _tpl(ctx, "build_defs.bzl", {
        "%{cm_is_configured}": "False",
    })

    genrules = [
    ]

    _tpl(ctx, "BUILD", {
        "%{cm_include_genrules}": "\n".join(genrules),
        "%{cm_headers}": '":cm-include",',
    })

def _create_cm_repository(ctx):
    _tpl(ctx, "build_defs.bzl", {
        "%{cm_is_configured}": "True",
    })

    genrule_name = "cm-include"
    src_dir = []
    src_dir.append("/usr/local/include/")
    command = []
    command.append("	mkdir include")
    command.append("	mkdir include/igfxcmrt")
    command.append("	cp /usr/local/include/igfxcmrt/cm_rt.h include/igfxcmrt/")
    command.append("	cp /usr/local/include/igfxcmrt/cm_rt.h $@")
    outs = []
    outs.append('	"include/igfxcmrt/cm_rt.h"')

    genrules = [
        #_genrule(src_dir, genrule_name, command, outs),
        #symlink_genrule_for_dir(ctx, "/usr/local/include/", "include", "cm-include"),
    ]

    _tpl(ctx, "BUILD", {
        "%{cm_include_genrules}": "\n".join(genrules),
        "%{cm_headers}": '":cm-include",',
    })

def _configure_cm_impl(ctx):
    _LIBVA_DRIVER_NAME = ctx.os.environ.get("LIBVA_DRIVER_NAME", "0").strip()
    _LIBVA_DRIVERS_PATH = ctx.os.environ.get("LIBVA_DRIVERS_PATH", "0").strip()
    _VAI_NEED_CM = ctx.os.environ.get("VAI_NEED_CM", "0").strip()

    if _LIBVA_DRIVER_NAME != 0 and ctx.path("%s" % _LIBVA_DRIVERS_PATH).exists:
        _create_cm_repository(ctx)
    else:
        _create_cm_repository(ctx)

configure_cm = repository_rule(
    environ = [
    ],
    implementation = _configure_cm_impl,
)
