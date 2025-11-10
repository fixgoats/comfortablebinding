--- Startup times for process: Primary (or UI client) ---

times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.002  000.002: --- NVIM STARTING ---
000.121  000.119: event init
000.196  000.075: early init
000.348  000.152: locale set
000.381  000.033: init first window
000.689  000.307: inits 1
000.696  000.008: window checked
000.698  000.002: parsing arguments
001.071  000.028  000.028: require('vim.shared')
001.132  000.027  000.027: require('vim.inspect')
001.161  000.023  000.023: require('vim._options')
001.164  000.090  000.040: require('vim._editor')
001.165  000.140  000.022: require('vim._init_packages')
001.166  000.328: init lua interpreter
002.961  001.795: nvim_ui_attach
003.146  000.185: nvim_set_client_info
003.147  000.001: --- NVIM STARTED ---

--- Startup times for process: Embedded ---

times in msec
 clock   self+sourced   self:  sourced script
 clock   elapsed:              other lines

000.001  000.001: --- NVIM STARTING ---
000.065  000.064: event init
000.122  000.057: early init
000.256  000.134: locale set
000.290  000.033: init first window
000.517  000.228: inits 1
000.526  000.009: window checked
000.528  000.002: parsing arguments
000.848  000.026  000.026: require('vim.shared')
000.928  000.025  000.025: require('vim.inspect')
000.962  000.027  000.027: require('vim._options')
000.964  000.111  000.059: require('vim._editor')
000.965  000.157  000.020: require('vim._init_packages')
000.966  000.282: init lua interpreter
001.003  000.037: expanding arguments
001.013  000.010: inits 2
001.196  000.183: init highlight
001.197  000.001: waiting for UI
001.292  000.095: done waiting for UI
001.297  000.005: clear screen
001.397  000.025  000.025: require('vim.keymap')
001.910  000.084  000.084: sourcing nvim_exec2()
001.982  000.683  000.574: require('vim._defaults')
001.983  000.003: init default mappings & autocommands
002.258  000.027  000.027: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/ftplugin.vim
002.284  000.013  000.013: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/indent.vim
002.339  000.019  000.019: require('config.keymaps')
002.706  000.267  000.267: require('lazy')
002.743  000.030  000.030: require('ffi')
002.766  000.011  000.011: require('vim.fs')
002.819  000.052  000.052: require('vim.uri')
002.826  000.081  000.018: require('vim.loader')
002.888  000.053  000.053: require('lazy.stats')
002.960  000.059  000.059: require('lazy.core.util')
003.018  000.057  000.057: require('lazy.core.config')
003.110  000.033  000.033: require('lazy.core.handler')
003.268  000.047  000.047: require('lazy.pkg')
003.271  000.115  000.068: require('lazy.core.meta')
003.274  000.164  000.049: require('lazy.core.plugin')
003.280  000.260  000.064: require('lazy.core.loader')
003.401  000.044  000.044: require('lazy.core.fragments')
004.022  000.124  000.124: require('lazy.core.handler.keys')
004.055  000.027  000.027: require('lazy.core.handler.cmd')
004.083  000.027  000.027: require('lazy.core.handler.event')
004.104  000.019  000.019: require('lazy.core.handler.ft')
004.229  000.055  000.055: sourcing nvim_exec2() called at /opt/nvim-linux-x86_64/share/nvim/runtime/filetype.lua:0
004.231  000.087  000.032: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/filetype.lua
004.233  000.099  000.012: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.265  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.270  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.330  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.337  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.447  000.021  000.021: sourcing /home/fixgoats/.local/share/nvim/lazy/vimtex/plugin/vimtex.vim
004.448  000.034  000.012: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.453  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.484  000.009  000.009: sourcing /home/fixgoats/.local/share/nvim/lazy/vimtex/ftdetect/cls.vim
004.486  000.021  000.012: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.504  000.008  000.008: sourcing /home/fixgoats/.local/share/nvim/lazy/vimtex/ftdetect/tex.vim
004.505  000.017  000.010: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.524  000.007  000.007: sourcing /home/fixgoats/.local/share/nvim/lazy/vimtex/ftdetect/tikz.vim
004.525  000.018  000.010: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.528  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.675  000.002  000.002: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
004.681  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
005.137  000.121  000.121: require('vim.lsp.log')
005.390  000.251  000.251: require('vim.lsp.protocol')
005.590  000.197  000.197: require('vim.lsp.util')
005.752  000.078  000.078: require('vim.lsp.sync')
005.760  000.167  000.089: require('vim.lsp._changetracking')
005.900  000.064  000.064: require('vim.lsp._transport')
005.912  000.150  000.086: require('vim.lsp.rpc')
005.974  001.230  000.342: require('vim.lsp')
006.328  000.353  000.353: require('blink.cmp')
006.546  000.049  000.049: require('blink.cmp.lib.async')
006.594  000.020  000.020: require('blink.cmp.config.utils')
006.616  000.020  000.020: require('blink.cmp.config.keymap')
006.692  000.043  000.043: require('blink.cmp.config.completion.keyword')
006.716  000.023  000.023: require('blink.cmp.config.completion.trigger')
006.740  000.024  000.024: require('blink.cmp.config.completion.list')
006.774  000.033  000.033: require('blink.cmp.config.completion.accept')
006.809  000.033  000.033: require('blink.cmp.config.completion.menu')
006.848  000.039  000.039: require('blink.cmp.config.completion.documentation')
006.870  000.020  000.020: require('blink.cmp.config.completion.ghost_text')
006.871  000.254  000.039: require('blink.cmp.config.completion')
006.897  000.026  000.026: require('blink.cmp.config.fuzzy')
006.956  000.058  000.058: require('blink.cmp.config.sources')
006.995  000.038  000.038: require('blink.cmp.config.signature')
007.034  000.038  000.038: require('blink.cmp.config.snippets')
007.055  000.020  000.020: require('blink.cmp.config.appearance')
007.080  000.024  000.024: require('blink.cmp.config.modes.cmdline')
007.101  000.020  000.020: require('blink.cmp.config.modes.term')
007.103  000.556  000.037: require('blink.cmp.config')
007.133  000.030  000.030: require('blink.cmp.lib.utils')
007.153  000.019  000.019: require('blink.cmp.lib.event_emitter')
007.157  000.827  000.173: require('blink.cmp.sources.lib')
007.169  002.455  000.046: sourcing /home/fixgoats/.local/share/nvim/lazy/blink.cmp/plugin/blink-cmp.lua
007.172  002.479  000.023: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
007.178  000.002  000.002: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
007.183  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
007.804  000.028  000.028: require('blink.cmp.config.modes.types')
008.089  000.046  000.046: require('blink.cmp.fuzzy.download.system')
008.127  000.119  000.073: require('blink.cmp.fuzzy.download.files')
008.128  000.145  000.026: require('blink.cmp.fuzzy.download.git')
008.130  000.193  000.048: require('blink.cmp.fuzzy.download')
008.231  000.079  000.079: require('vim._system')
011.214  001.264  001.264: require('vim.filetype')
011.605  000.061  000.061: require('luasnip.util.types')
011.610  000.113  000.052: require('luasnip.util.ext_opts')
012.010  000.161  000.161: require('luasnip.util.lazy_table')
012.156  000.144  000.144: require('luasnip.extras.filetype_functions')
012.195  000.464  000.158: require('luasnip.default_config')
012.197  000.586  000.123: require('luasnip.session')
012.201  000.869  000.169: require('luasnip.config')
012.227  002.322  000.190: sourcing /home/fixgoats/.local/share/nvim/lazy/LuaSnip/plugin/luasnip.lua
012.232  002.352  000.029: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
012.275  000.018  000.018: sourcing /home/fixgoats/.local/share/nvim/lazy/LuaSnip/plugin/luasnip.vim
012.277  000.037  000.019: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
012.282  000.002  000.002: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
012.294  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
012.478  000.036  000.036: require('vim.version')
013.253  000.839  000.802: require('luasnip.util.vimversion')
013.258  000.891  000.053: require('luasnip.util.util')
013.459  000.063  000.063: require('luasnip.util.path')
013.525  000.022  000.022: require('luasnip.session.snippet_collection.source')
013.577  000.021  000.021: require('luasnip.util.table')
013.598  000.020  000.020: require('luasnip.util.auto_table')
013.602  000.141  000.078: require('luasnip.session.snippet_collection')
013.667  000.064  000.064: require('luasnip.util.log')
013.672  000.412  000.145: require('luasnip.loaders.util')
013.730  000.055  000.055: require('luasnip.loaders.fs_watchers')
013.753  000.003  000.003: require('vim.F')
013.756  000.025  000.022: require('luasnip.loaders.data')
013.775  000.019  000.019: require('luasnip.session.enqueueable_operations')
014.128  000.179  000.179: require('luasnip.util.str')
014.155  000.024  000.024: require('luasnip.nodes.key_indexer')
014.179  000.023  000.023: require('luasnip.util.feedkeys')
014.231  000.051  000.051: require('luasnip.nodes.util.snippet_string')
014.235  000.324  000.047: require('luasnip.nodes.util')
014.269  000.033  000.033: require('luasnip.util.events')
014.314  000.043  000.043: require('luasnip.nodes.optional_arg')
014.325  000.450  000.050: require('luasnip.nodes.node')
014.400  000.022  000.022: require('luasnip.util.extend_decorator')
014.408  000.082  000.060: require('luasnip.nodes.insertNode')
014.437  000.028  000.028: require('luasnip.nodes.textNode')
014.468  000.030  000.030: require('luasnip.util.mark')
014.550  000.024  000.024: require('luasnip.util.select')
014.569  000.018  000.018: require('luasnip.util.time')
014.786  000.290  000.248: require('luasnip.util._builtin_vars')
014.809  000.340  000.050: require('luasnip.util.environ')
014.844  000.034  000.034: require('luasnip.util.pattern_tokenizer')
014.868  000.023  000.023: require('luasnip.util.dict')
015.247  000.355  000.355: require('luasnip.util.jsregexp')
015.251  000.381  000.027: require('luasnip.nodes.util.trig_engines')
015.304  001.505  000.138: require('luasnip.nodes.snippet')
015.408  000.027  000.027: require('luasnip.util.parser.neovim_ast')
015.615  000.201  000.201: require('luasnip.util.jsregexp')
015.658  000.041  000.041: require('luasnip.util.directed_graph')
015.661  000.313  000.044: require('luasnip.util.parser.ast_utils')
015.704  000.042  000.042: require('luasnip.nodes.functionNode')
015.813  000.107  000.107: require('luasnip.nodes.choiceNode')
015.903  000.088  000.088: require('luasnip.nodes.dynamicNode')
015.938  000.034  000.034: require('luasnip.util.functions')
015.943  000.638  000.054: require('luasnip.util.parser.ast_parser')
016.057  000.113  000.113: require('luasnip.util.parser.neovim_parser')
016.062  002.285  000.030: require('luasnip.util.parser')
016.120  000.057  000.057: require('luasnip.nodes.snippetProxy')
016.251  000.128  000.128: require('luasnip.util.jsonc')
016.315  000.032  000.032: require('luasnip.nodes.duplicate')
016.318  000.065  000.033: require('luasnip.loaders.snippet_cache')
016.325  006.827  000.497: require('luasnip.loaders.from_vscode')
018.060  000.069  000.069: require('luasnip.nodes.multiSnippet')
019.062  000.074  000.074: sourcing /home/fixgoats/.local/share/nvim/lazy/conform.nvim/plugin/conform.lua
019.066  000.098  000.024: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
019.072  000.002  000.002: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
019.083  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
019.461  000.373  000.373: require('conform')
019.684  000.003  000.003: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
019.713  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
019.918  000.002  000.002: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
019.927  000.001  000.001: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
020.042  000.055  000.055: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/editorconfig.lua
020.045  000.070  000.015: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
020.176  000.115  000.115: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/gzip.vim
020.178  000.127  000.012: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
020.250  000.059  000.059: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/man.lua
020.252  000.071  000.012: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
021.670  000.130  000.130: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/pack/dist/opt/matchit/plugin/matchit.vim
021.687  001.421  001.291: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/matchit.vim
021.691  001.435  000.013: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
021.800  000.082  000.082: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/matchparen.vim
021.802  000.095  000.013: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.192  000.148  000.148: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/pack/dist/opt/netrw/plugin/netrwPlugin.vim
022.202  000.354  000.206: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/netrwPlugin.vim
022.203  000.368  000.014: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.284  000.056  000.056: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/osc52.lua
022.286  000.071  000.015: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.377  000.076  000.076: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/rplugin.vim
022.378  000.089  000.014: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.425  000.028  000.028: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/shada.vim
022.427  000.042  000.013: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.451  000.008  000.008: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/spellfile.vim
022.453  000.023  000.015: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.513  000.046  000.046: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/tarPlugin.vim
022.515  000.058  000.012: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.568  000.036  000.036: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/tohtml.lua
022.569  000.050  000.014: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.598  000.009  000.009: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/tutor.vim
022.600  000.022  000.013: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.684  000.068  000.068: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/plugin/zipPlugin.vim
022.685  000.080  000.012: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.791  000.010  000.010: sourcing nvim_exec2() called at /home/fixgoats/.local/share/nvim/lazy/night-owl.nvim/after/plugin/autocmds.lua:0
022.793  000.040  000.030: sourcing /home/fixgoats/.local/share/nvim/lazy/night-owl.nvim/after/plugin/autocmds.lua
022.795  000.059  000.019: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
022.823  020.483  006.421: require('config.lazy')
023.394  000.138  000.138: require('thorn')
023.420  000.022  000.022: require('thorn.highlights')
023.459  000.038  000.038: require('thorn.colors')
023.514  000.053  000.053: require('thorn.groups')
023.567  000.030  000.030: require('thorn.groups.kinds')
023.628  000.035  000.035: require('thorn.groups.snacks')
023.668  000.022  000.022: require('thorn.groups.bufferline')
023.688  000.018  000.018: require('thorn.groups.gitsigns')
023.708  000.019  000.019: require('thorn.groups.trouble')
023.758  000.048  000.048: require('thorn.groups.base')
023.799  000.024  000.024: require('thorn.groups.semantic_tokens')
023.850  000.024  000.024: require('thorn.groups.nvim_tree')
023.874  000.021  000.021: require('thorn.groups.cmp')
023.923  000.022  000.022: require('thorn.groups.lazy')
023.947  000.022  000.022: require('thorn.groups.render_markdown')
024.012  000.024  000.024: require('thorn.groups.telescope')
024.057  000.042  000.042: require('thorn.groups.treesitter')
024.645  001.410  000.809: sourcing /home/fixgoats/.local/share/nvim/lazy/thorn.nvim/colors/thorn.lua
024.668  001.786  000.376: sourcing nvim_exec2() called at /home/fixgoats/.config/nvim/init.lua:0
024.671  001.846  000.061: require('config.looks')
024.862  000.132  000.132: require('vim.diagnostic')
024.866  000.194  000.062: require('config.lsp')
024.867  022.559  000.016: sourcing /home/fixgoats/.config/nvim/init.lua
024.871  000.289: sourcing vimrc file(s)
024.987  000.060  000.060: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/filetype.lua
025.186  000.044  000.044: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/syntax/synload.vim
025.239  000.207  000.163: sourcing /opt/nvim-linux-x86_64/share/nvim/runtime/syntax/syntax.vim
025.248  000.110: inits 3
026.210  000.962: reading ShaDa
026.338  000.063  000.063: require('luasnip.loaders')
026.392  000.042  000.042: require('luasnip.loaders.from_lua')
026.433  000.038  000.038: require('luasnip.loaders.from_snipmate')
026.478  000.126: opening buffers
026.493  000.014: BufEnter autocommands
026.494  000.002: editing files in windows
026.546  000.051: VimEnter autocommands
026.598  000.029  000.029: require('vim.termcap')
026.612  000.008  000.008: require('vim.text')
026.625  000.042: UIEnter autocommands
026.626  000.001: before starting main loop
026.808  000.182: first screen update
026.809  000.001: --- NVIM STARTED ---

