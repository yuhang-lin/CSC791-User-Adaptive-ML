# server = Sys.info()['nodename']
# if (server == 'server1') {
#     dnroot = 'folder1'
# }else if (grepl('domain2', server)) {
#     dnroot = 'folder2'
# }else if (grepl('domain3', server)) {
#     dnroot = 'folder3'
# }else {
#     quit(sprintf('unrecognized server %s\n', server))
# }

dnroot = '~/Downloads/data/new_synth/train'

tests = c('X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13')
fnlog.tmp = sprintf('%s/micegp_log/%%s_output_iter%%s.RData', dnroot)
fnres.tmp = sprintf('%s/micegp_log/%%s_res_iter%%s.RData', dnroot)
fnkm.tmp = sprintf('%s/micegp_log/%%s_km_iter%%s.RData', dnroot)

timeidx = 'time'


