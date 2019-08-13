def _tpl(ctx, tpl, substitutions = {}, out = None):
    if not out:
        out = tpl.replace(":", "/")
    ctx.template(
        out,
        Label("@com_intel_plaidml//vendor/cm:%s.tpl" % tpl),
        substitutions,
    )

def _create_cm_dummy_repository(ctx):
    _tpl(ctx, "build_defs.bzl", {
        "%{cm_is_configured}": "False",
        "%{cm_runtime_lib}": "''",
    })

    genrules = [
    ]

    _tpl(ctx, "BUILD", {
        "%{cm_include_genrules}": "\n".join(genrules),
        "%{cm_headers}": '":cm-include",',
    })

def _create_cm_repository(ctx):
    _CM_RUNTIME_LIB = ctx.os.environ.get("CM_RUNTIME_LIB", "0").strip()

    _tpl(ctx, "build_defs.bzl", {
        "%{cm_is_configured}": "True",
        "%{cm_runtime_lib}": "'" + _CM_RUNTIME_LIB + "'",
    })

    genrules = [
    ]

    _tpl(ctx, "BUILD", {
        "%{cm_include_genrules}": "\n".join(genrules),
        "%{cm_headers}": '":cm-include",',
    })

def _configure_cm_impl(ctx):
    _LIBVA_DRIVER_NAME = ctx.os.environ.get("LIBVA_DRIVER_NAME", "0").strip()
    _LIBVA_DRIVERS_PATH = ctx.os.environ.get("LIBVA_DRIVERS_PATH", "0").strip()
    _CM_RUNTIME_LIB = ctx.os.environ.get("CM_RUNTIME_LIB", "0").strip()
    _VAI_NEED_CM = ctx.os.environ.get("VAI_NEED_CM", "0").strip()

    if _VAI_NEED_CM == "1" and _CM_RUNTIME_LIB != 0 and _LIBVA_DRIVER_NAME != 0 and ctx.path("%s" % _LIBVA_DRIVERS_PATH).exists:
        _create_cm_repository(ctx)
    else:
        _create_cm_dummy_repository(ctx)

configure_cm = repository_rule(
    environ = [
    ],
    implementation = _configure_cm_impl,
)
