"""
for greatlakes:
PART=standard; CAP=$(sinfo -N -h -p "$PART" -o '%N|%c' | sort -t'|' -k1,1 -u | awk -F'|' '{s+=$2} END{print s+0}'); squeue -r -h -p "$PART" -t PD -o '%C|%l' | awk -F'|' -v cap="$CAP" 'function hrs(t,a,n,d,day){if(t=="UNLIMITED"||t=="NOT_SET"||t=="N/A")return 0;d=0;day=0;if(index(t,"-")){split(t,a,"-");d=a[1];t=a[2];day=1}n=split(t,a,":");if(day){if(n==3)return 24*d+a[1]+a[2]/60+a[3]/3600;if(n==2)return 24*d+a[1]+a[2]/60;return 24*d+a[1]}if(n==3)return a[1]+a[2]/60+a[3]/3600;if(n==2)return a[1]/60+a[2]/3600;return a[1]/60}{q+=$1*hrs($2)}END{if(cap>0)printf "Pending CPU-hours: %.1f | Standard CPUs: %d | Crude backlog: %.2f hours = %.2f days\n",q,cap,q/cap,q/cap/24;else print "No standard-partition CPUs visible"}'

for raven:
CAP=$(sinfo -N -h -o '%N|%G' | awk -F'|' 'tolower($2)!~/gpu/ && !seen[$1]++ {n+=72} END{print n+0}'); squeue -r -h -t PD -o '%C|%l|%P' | awk -F'|' -v cap="$CAP" 'function hrs(t,a,n,d,hd){if(t=="UNLIMITED"||t=="NOT_SET")return 0;d=0;hd=0;if(index(t,"-")){split(t,a,"-");d=a[1];t=a[2];hd=1}n=split(t,a,":");if(hd){if(n==3)return 24*d+a[1]+a[2]/60+a[3]/3600;if(n==2)return 24*d+a[1]+a[2]/60;return 24*d+a[1]}if(n==3)return a[1]+a[2]/60+a[3]/3600;if(n==2)return a[1]/60+a[2]/3600;return a[1]/60}tolower($3)!~/gpu|interactive/{q+=$1*hrs($2)}END{if(cap>0)printf "Pending CPU-hours: %.1f | Raven physical cores: %d | Crude backlog: %.2f hours = %.2f days\n",q,cap,q/cap,q/cap/24;else print "Could not determine CPU capacity"}'

for viper:
CAP=$(sinfo -N -h -o '%N|%c' | sort -t'|' -k1,1 -u | awk -F'|' '{s+=$2} END{print s+0}'); squeue --array -h -t PD -o '%C|%l' | awk -F'|' -v cap="$CAP" 'function hrs(t,a,n,d,hd){if(t=="UNLIMITED"||t=="NOT_SET")return 0;d=0;hd=0;if(index(t,"-")){split(t,a,"-");d=a[1];t=a[2];hd=1}n=split(t,a,":");if(hd){if(n==3)return 24*d+a[1]+a[2]/60+a[3]/3600;if(n==2)return 24*d+a[1]+a[2]/60;return 24*d+a[1]}if(n==3)return a[1]+a[2]/60+a[3]/3600;if(n==2)return a[1]/60+a[2]/3600;return a[1]/60}{q+=$1*hrs($2)}END{if(cap>0)printf "Pending CPU-hours: %.1f | Cluster CPUs: %d | Crude backlog: %.2f hours = %.2f days\n",q,cap,q/cap,q/cap/24;else print "Could not determine CPU count"}'
"""

