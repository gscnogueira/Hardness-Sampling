get_running_info(){
	local log_file=$1
	local main_pid=$(grep Main $log_file | awk -F ' - ' '{print $2}')
	
	pgrep -P "$main_pid" | while read -r child_pid; do
		grep "$child_pid - .* - INFO" "$log_file" | tail -1
	
	done

}

get_running_learners(){
	local log_file=$1
	get_running_info "$log_file" | awk -F '[(,)]' '{print $4}' | sort | uniq -c | sort -nr
}

get_running_methods(){
	local log_file=$1
	get_running_info "$log_file" | awk -F '[(,)]' '{print $5}' | sort | uniq -c | sort -nr
}


running(){
	local log_file=$1
	get_running_info "$log_file" | awk -F '[(,)]' '{print $4, $5}' | sort | uniq -c | sort -nr
}


finished(){
	# Mostra quantidade de experimentos para os pares learner-método que tiveram a execução finalizada
	local log_file=$1
	grep -i finalizado "$log_file" | awk -F '[(,)]' '{print $4, $5}'| sort | uniq -c | sort -nr
}

log_report(){
    local log_file=$1
    # Cores ANSI (para terminal)
    local RED='\033[0;31m'
    local GREEN='\033[0;32m'
    local BLUE='\033[0;34m'
    local NC='\033[0m' # No Color
    
    # Função para criar cabeçalho centralizado
    print_header() {
        local title=$1
        local color=$2
        local width=60
        local padding=$(( ($width - ${#title} - 2) / 2 ))
        printf "${color}"
	printf "$title:\n"
        printf "${NC}"
    }

    # Relatório de processos running
    print_header " RUNNING " "$BLUE"
    running "$log_file" | column 
    
    # Espaçamento entre seções
    echo -e "\n"
    
    # Relatório de processos finished
    print_header " FINISHED " "$GREEN"
    finished "$log_file" | column
    
}


